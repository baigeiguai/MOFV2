import torch 
from models.TransformerEncoder import TransformerEncoder

class RetryViT(torch.nn.Module):
    def __init__(self,patch_size=20,seq_len=425,num_classes=230,dim=128,depth=32,heads=4,dropout=0,in_c=2) -> None:
        super(RetryViT,self).__init__()
        self.patch_size = patch_size
        self.seq_len = seq_len
        self.project = torch.nn.Linear(patch_size*in_c,dim)
        self.transformers = TransformerEncoder(seq_len,dim,depth,heads,dropout,dim*4)
        self.cls = torch.nn.Linear(dim,num_classes)
    def forward(self,intensity,angle):
        intensity = intensity.view(intensity.shape[0],-1,1).type(torch.float)
        angle = angle.view(angle.shape[0],-1,1).type(torch.float)
        x = torch.concat([angle,intensity],dim=-1)
        x = x.view(x.shape[0],self.seq_len,-1)
        x = self.project(x)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.cls(x)
        return x 