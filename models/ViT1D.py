from sympy import im
import torch
from torch import nn

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, seq_len, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert (seq_len % patch_size) == 0

        num_patches = seq_len // patch_size
        patch_dim = channels * patch_size

        # self.resize = nn.Sequential(
        #     Rearrange('b c (n p) -> b n (p c)', p = patch_size),
        #     )

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p = patch_size),
            nn.Linear(patch_dim, dim),
        )
        
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)



        self.mlp_cls = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.mlp_token = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,dim),
            nn.LeakyReLU(),
            nn.Linear(dim,dim),
            nn.LeakyReLU(),
            nn.Linear(dim,patch_dim),
            Rearrange('b n (p c) -> b c (n p)',p=patch_size),
        )

    def forward(self, series):
        x = self.to_patch_embedding(series)
        b,n,_  = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)

        x, ps = pack([cls_tokens, x], 'b * d')

        # x += self.pos_embedding[:, :(n + 1)]

        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return self.mlp_cls(cls_tokens)


class ViTDecoder(nn.Module):
    def __init__(self,depth,dim,heads,mlp_dim,dropout,channels,patchsize ) -> None:
        super(ViTDecoder,self).__init__()
        self.transformer = Transformer(dim=dim,depth=depth,heads=heads,dim_head=dim//heads,mlp_dim=mlp_dim,dropout=dropout)
        self.to_patch_embedding =  nn.Sequential(
                nn.Linear(dim,channels*patchsize),
                Rearrange('b n (p c) -> b c (n p)',p=patchsize)
            )

    def forward(self,x,useless=True):
        x = self.transformer(x)
        x = self.to_patch_embedding(x)
        return x

if __name__ == '__main__':

    device  = torch.device('cuda:3')
    v = ViT(
        seq_len = 8500,
        patch_size = 25,
        num_classes = 230,
        dim = 128,
        depth = 32,
        heads = 1,
        mlp_dim = 256,
        dropout = 0.05,
        emb_dropout = 0.05,
        dim_head=16,
        channels=2 
    ).to(device)

    # d = ViTDecoder(8,256,8,512,0.05,2,25).to(device)

    time_series = torch.randn((256,2,8500)).to(device)
    logits = v(time_series)
    print(logits.shape)
    
    from torchinfo import summary
    summary(v,(256,2,8500))
    # raw = d(logits)
    # print(raw.shape)
    # model = torch.nn.Sequential(
    #     v,
    #     d        
    # )
    # print(model(time_series).shape)
    # a,b = v(time_series,True)
    # print(a.shape,b.shape)