import torch 
from models.ResTcn import ResTcn

class ConcatEmbedConv(torch.nn.Module):
    def __init__(self,embed_len=32):
        super(ConcatEmbedConv,self).__init__()
        self.embed = torch.nn.Embedding(8500,embed_len)
        self.conv = ResTcn(embed_len+1)
        
    def forward(self,intensity,angle):
        angle = angle.view(angle.shape[0],-1).type(torch.long)
        intensity = intensity.view(intensity.shape[0],-1,1)/100 
        x = self.embed(angle)
        x = torch.concat([x,intensity],dim=-1)
        return self.conv(x)