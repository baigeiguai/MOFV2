import torch 
from models.BiMamba import BiMamba,MambaConfig
from models.ResTcn import ResTcn

class ExplorerV1(torch.nn.Module):
    def __init__(self,d_state,n_layers,n_class=230,embed_len=1024) -> None:
        super(ExplorerV1,self).__init__()
        config = MambaConfig(2,n_layers=n_layers,d_state=d_state)
        self.predict_hkl_block = BiMamba(config)
        # self.project = torch.nn.Linear(2,3)
        # self.conv = ResTcn(3)
        # self.conv = ResTcn(2)
        
        
    
    def forward(self,intensity,angle):
        intensity = intensity.view(intensity.shape[0],-1,1)
        angle = angle.view(angle.shape[0],-1,1)
        data = torch.concat([intensity,angle],dim=-1)
        # hkl = self.project(self.predict_hkl_block(data))
        hkl = self.predict_hkl_block(data)
        return hkl
        return self.conv(hkl)        
        return  [self.conv(hkl),hkl]