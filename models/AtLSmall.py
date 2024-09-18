import torch
from models.TransformerEncoder import TransformerEncoder

MIN_ANGLE = 5
MAX_ANGLE = 90
FIXED_1D_LENGTH = 8500

class AtLSmall(torch.nn.Module):
    def __init__(self,patch_len=10,embed_len=32,n_layers=2,p_dropout=0,d_ff=512):
        super(AtLSmall,self).__init__()
        self.patch_len = patch_len
        self.embed = torch.nn.Embedding(FIXED_1D_LENGTH+1,embed_len-1,padding_idx=0)
        self.att = TransformerEncoder(FIXED_1D_LENGTH//patch_len,patch_len*embed_len,n_layers,p_drop=p_dropout,d_ff=d_ff)
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(320,256),
            torch.nn.Linear(256,230),
        )
    def forward(self,intensity,angle):
        angle = angle.view(angle.shape[0],-1).type(torch.float)
        att_not_mask = angle > 1e-5
        angle_index = (angle-MIN_ANGLE)*(FIXED_1D_LENGTH//(MAX_ANGLE-MIN_ANGLE))*(angle>1e-5)
        angle_index = angle_index.type(torch.long)
        
        angle_features = self.embed(angle_index)
        intensity = intensity.view(intensity.shape[0],-1,1).type(torch.float)
        x = torch.concat([intensity,angle_features],dim=-1)
        x = x.view(x.shape[0],x.shape[1]//self.patch_len,-1)
        att_not_mask = att_not_mask.view(att_not_mask.shape[0],-1,self.patch_len).max(dim=-1).values
        att_not_mask = att_not_mask.view(att_not_mask.shape[0],1,-1)
        att_mask = att_not_mask<1e-5
        x= self.att(x,att_mask)
        x=x.mean(dim=1)
        x = self.cls(x)
        return x