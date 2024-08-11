import torch 
from models.ResTcnSmall import ResTcn
from mamba_ssm import Mamba

class ExplorerV2(torch.nn.Module):
    def __init__(self,d_model,d_state,n_layers,conv_p_dropout) -> None:
        super(ExplorerV2,self).__init__()
        self.predict_hkl_blocks = torch.nn.ModuleList([Mamba(d_model=d_model,d_state=d_state) for _ in range(n_layers)])
        self.conv = ResTcn(2,conv_p_dropout)
        self.intensity_norm = torch.nn.BatchNorm1d(1)
    
    def forward(self,angle,intensity):
        angle = angle.view(angle.shape[0],-1,1)
        intensity = intensity.view(intensity.shape[0],1,-1)
        intensity = self.intensity_norm(intensity)
        intensity = intensity.view(intensity.shape[0],-1,1)
        x = torch.concat([angle.deg2rad().sin(),intensity],dim=-1)
        # x = x.transpose(-1,-2)
        for block in self.predict_hkl_blocks:
            x = block(x)
        x = x.transpose(-1,-2)
        return self.conv(x)