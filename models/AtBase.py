import torch

from models.TransformerEncoder import EncoderLayer

ANGLE_NUM = 850 

# class AttentionBlock(torch.nn.Module):
    
#     def forward(self,)

class AtBase(torch.nn.Module):
    def __init__(self,embed_len=64,n_layers=24,n_heads=4,p_drop=0.1,d_ff=256):
        super(AtBase,self).__init__()
        self.embed = torch.nn.Embedding(ANGLE_NUM,embed_len)
        self.transformer_encoders =  torch.nn.ModuleList([EncoderLayer(embed_len,n_heads,p_drop,d_ff) for _ in range(n_layers)])        
        self.cls = torch.nn.Linear(embed_len*ANGLE_NUM,230)
        
    def forward(self,intensity,angle):
        angle = angle.view(angle.shape[0],-1).type(torch.long)
        intensity = intensity.view(intensity.shape[0],-1,1)/100
        x = self.embed(angle)
        x = x * intensity
        for encoder in self.transformer_encoders:
            x = encoder(x)
        x = x.view(x.shape[0],-1)
        return self.cls(x) 
        