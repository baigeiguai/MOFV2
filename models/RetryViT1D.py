import torch 
from models.ViT1D import ViT

class RetryViT(torch.nn.Module):
    def __init__(self,patch_size=25,num_classes=230,dim=128,depth=32,heads=1,mlp_dim=256,dropout=0.05,emb_dropout = 0.05,dim_head=16,channels=2) -> None:
        super(RetryViT,self).__init__()
        self.ViT = ViT(
        seq_len = 8500,
        patch_size = patch_size,
        num_classes = num_classes,
        dim = dim,
        depth = depth,
        heads = heads,
        mlp_dim = mlp_dim,
        dropout = dropout,
        emb_dropout = emb_dropout,
        dim_head=dim_head,
        channels=channels )
        
    def forward(self,intensity,angle):
        intensity = intensity.view(intensity.shape[0],1,-1)
        angle = angle.view(angle.shape[0],1,-1)
        x = torch.concat([angle.deg2rad().sin(),intensity/100],dim=1)
        return self.ViT(x)