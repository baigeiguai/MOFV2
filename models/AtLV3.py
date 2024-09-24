import torch 
from models.TransformerEncoder import TransformerEncoder

MIN_ANGLE = 5
MAX_ANGLE = 90
FIXED_1D_LENGTH = 850

class AtLV3(torch.nn.Module):
    def __init__(self,embed_len=64,n_layers=16,p_dropout=0,d_ff=512):
        super(AtLV3,self).__init__()
        self.embed = torch.nn.Embedding(FIXED_1D_LENGTH+1,embed_len-1)
        self.att = TransformerEncoder(FIXED_1D_LENGTH,dimension=embed_len,n_layers=n_layers,p_drop=p_dropout,d_ff=d_ff)
    
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(embed_len,128),
            torch.nn.Linear(128,230)   
        )
    def forward(self,intensity,angle):
        intensity = intensity.view(intensity.shape[0],-1).type(torch.float)
        att_mask = (intensity<1e-6).view(intensity.shape[0],1,-1)
        angle = angle.view(angle.shape[0],-1).type(torch.float)
        angle_index = (angle-MIN_ANGLE)*(FIXED_1D_LENGTH//(MAX_ANGLE-MIN_ANGLE))
        angle_index = angle_index.type(torch.long)
        angle_features = self.embed(angle_index)
        intensity = intensity.view(intensity.shape[0],-1,1)
        
        x = torch.concat([intensity,angle_features],dim=-1)
        x = self.att(x,att_mask)
        x = x.max(dim=1).values
        x = self.cls(x)
        return x 