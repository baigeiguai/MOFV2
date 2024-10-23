import torch 
from models.HopeV1 import AttentionModule

class HopeV1AttOnly(torch.nn.Module):
    def __init__(self):
        super(HopeV1AttOnly,self).__init__()
        self.att = AttentionModule(embed_len=47)
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(64,128),
            torch.nn.Linear(128,230),
        )
    def forward(self,intensity,angle):
        x = self.att(intensity,angle)
        x = self.cls(x)
        return x 