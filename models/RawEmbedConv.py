import torch 
from models.ResTcn import ResTcn

class RawEmbedConv(torch.nn.Module):
    def __init__(self,embed_len=32):
        super(RawEmbedConv,self).__init__()
        self.embed = torch.nn.Embedding(8500,embed_len)
        self.conv = ResTcn(embed_len,)
        
    def forward(self,intensity,angle):
        angle = angle.view(angle.shape[0],-1).type(torch.long)
        intensity = intensity.view(intensity.shape[0],-1,1)/100 
        x = self.embed(angle)
        x = x.transpose(1,2)
        intensity = intensity.transpose(1,2)
        x *= intensity
        return self.conv(x)
        