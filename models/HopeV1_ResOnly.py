import torch 

from models.ResTcn_8500 import ResTcn

class HopeV1ResOnly(torch.nn.Module):
    def __init__(self,p_dropout=0.2):
        super(HopeV1ResOnly,self).__init__()
        self.conv = ResTcn(2,p_dropout)
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(1024,512),
            torch.nn.Linear(512,230)
        )
    def forward(self,intensity,angle):
        x = self.conv(intensity,angle)        
        return self.cls(x)