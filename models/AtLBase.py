import torch
import numpy as np 

class Attention(torch.nn.Module):
    def __init__(self,dimension):
        super(Attention,self).__init__()
        self.d_k = self.d_v = dimension

        self.WQ = torch.nn.Linear(dimension,dimension)
        self.WK = torch.nn.Linear(dimension,dimension)
        self.WV = torch.nn.Linear(dimension,dimension)
        self.linear = torch.nn.Linear(self.d_v,dimension)

    def forward(self,X,intensity):
        batch_size = X.shape[0]
        q = self.WQ(X)
        k_T = self.WK(X).transpose(-1,-2)
        v = self.WV(X)
        atten_score = q@(k_T/np.sqrt(self.d_k))
        attn_weights = atten_score.softmax(dim=-1)+intensity
        out = attn_weights@v 
        return self.linear(out)


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self,dimension,d_ff) -> None:
        super(FeedForwardNetwork,self).__init__()
        self.linear1 = torch.nn.Linear(dimension,d_ff)
        self.linear2 = torch.nn.Linear(d_ff,dimension)
        self.relu = torch.nn.LeakyReLU()

    def forward(self,X):
        return self.linear2(self.relu(self.linear1(X)))

class EncoderLayer(torch.nn.Module):
    def __init__(self,dimension,p_dropout,d_ff) -> None:
        super(EncoderLayer,self).__init__()
        self.att = Attention(dimension)
        
        self.dropout1 = torch.nn.Dropout(p_dropout)
        self.layernorm1 = torch.nn.LayerNorm(dimension,eps=1e-6)
        
        self.ffn = FeedForwardNetwork(dimension,d_ff)

        self.dropout2 = torch.nn.Dropout(p_dropout)

        self.layernorm2 = torch.nn.LayerNorm(dimension,eps=1e-6)

    def forward(self,X,intensity):
        # print("[EncoderLayer]X.shape:",X.shape)
        attn_output = self.att(X,intensity)
        
        attn_output = self.dropout1(attn_output)
        # print("[EncoderLayer]X.shape:",attn_output.shape)
        attn_output = self.layernorm1(attn_output+X)
        # print("[EncoderLayer]X.shape:",attn_output.shape)
        ffn_output = self.ffn(attn_output)
        # print("[EncoderLayer]X.shape:",ffn_output.shape)
        ffn_output = self.dropout2(ffn_output)
        # print("[EncoderLayer]X.shape:",ffn_output.shape)
        ffn_output = self.layernorm2(ffn_output+attn_output)
        # print("[EncoderLayer]X.shape:",ffn_output.shape)
        return ffn_output


ANGLE_NUM =850


class AtLBase(torch.nn.Module):
    def __init__(self,embed_len=64,n_layers=24,p_drop=0.05,d_ff=256):
        super(AtLBase,self).__init__()
        self.embed = torch.nn.Embedding(ANGLE_NUM,embed_len)
        self.encoders =  torch.nn.ModuleList([EncoderLayer(embed_len,p_drop,d_ff) for _ in range(n_layers)])        
        self.cls = torch.nn.Linear(embed_len*ANGLE_NUM,230)
        
    def forward(self,intensity,angle):
        angle = angle.view(angle.shape[0],-1).type(torch.long)
        intensity = torch.diag_embed(intensity.view(intensity.shape[0],-1))
        x = self.embed(angle)
        for encoder in self.encoders:
            x = encoder(x,intensity)
        x = x.view(x.shape[0],-1)
        return self.cls(x)
            
        
        
    
    