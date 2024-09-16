import torch 
from models.TransformerEncoder import TransformerEncoder

MIN_ANGLE = 5
MAX_ANGLE = 90
FIXED_1D_LENGTH = 8500

class AtLV2(torch.nn.Module):
    def __init__(self,embed_len=64,seq_len=FIXED_1D_LENGTH,n_layers=8,p_drop=0,d_ff=256):
        super(AtLV2,self).__init__()
        self.embed = torch.nn.Embedding(FIXED_1D_LENGTH,embed_len-1,padding_idx=0)
        self.att = TransformerEncoder(seq_len=seq_len,dimension=embed_len,n_layers=n_layers,p_drop=p_drop,d_ff=d_ff)
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(64,128),
            torch.nn.Linear(128,230),            
        )

    def forward(self,intensity,angle):
        att_mask = angle.view(angle.shape[0],1,-1)<1e-5
        angle = angle.view(angle.shape[0],-1).type(torch.float)
        angle_index = (angle-MIN_ANGLE)*(FIXED_1D_LENGTH//(MAX_ANGLE-MIN_ANGLE))*(angle>1e-5)
        angle_index = angle_index.type(torch.long)
        print(angle_index.max(),angle_index.min())
        angle_features = self.embed(angle_index)
        intensity = intensity.view(intensity.shape[0],-1,1)
        x = torch.concat([intensity,angle_features],dim=-1)
        x = self.att(x,att_mask)
        x = x.mean(dim=1)
        x = self.cls(x)
        return x 