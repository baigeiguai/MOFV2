import torch 
from models.ResTcn import ResTcn

ANGLE_NUM = 8500

class XrdAttentionBase(torch.nn.Module):
    def __init__(self,embed_len=32) -> None:
        super(XrdAttentionBase,self).__init__()
        self.embed = torch.nn.Embedding(ANGLE_NUM,embed_len)
        self.conv = ResTcn(embed_len)
    
    def forward(self,intensity,angle):
        angle = angle.view(angle.shape[0],-1).type(torch.long)
        intensity = intensity.view(intensity.shape[0],-1,1)/100
        intensity_T = intensity.transpose(1,2)
        atten = (intensity@intensity_T).softmax(dim=-1)
        x = self.embed(angle)
        x = atten@x
        x = x.transpose(1,2)
        return self.conv(x)
