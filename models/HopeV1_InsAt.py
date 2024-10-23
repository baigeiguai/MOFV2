import torch 
from models.HopeV1 import HopeV1
from models.TransformerEncoder import TransformerEncoder

class HopeV1InsAt(torch.nn.Module):
    def __init__(self,batch_size,embed_len=1056,n_layers=4,p_dropout=0.1,d_ff=1024):
        super(HopeV1InsAt,self).__init__()
        self.f = HopeV1()
        self.batch_size = batch_size
        self.pad = torch.nn.Parameter(torch.zeros(batch_size,embed_len),requires_grad=False)
        self.zeros = torch.nn.Parameter(torch.zeros(batch_size),requires_grad=False)
        self.ones = torch.nn.Parameter(torch.ones(batch_size),requires_grad=False)
        self.att = TransformerEncoder(batch_size,embed_len,n_layers,p_dropout,d_ff)
        self.cls = torch.nn.Sequential(   
            torch.nn.Linear(embed_len,512),
            torch.nn.Linear(512,230),
        )
    def forward(self,intensity,angle):
        x = self.f(intensity,angle)
        b = x.shape[0]
        pad_len = self.batch_size-b
        x = torch.concat([x,self.pad[:pad_len]],dim=0).view(1,self.batch_size,-1)
        masked = torch.concat([self.zeros[:b],self.ones[:pad_len]],dim=0).view(1,1,-1)        
        x = self.att(x).view(self.batch_size,-1)
        x = self.cls(x[:b])
        return x