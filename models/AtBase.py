import torch

from models.TransformerEncoder import EncoderLayer

class ResBlock1D(torch.nn.Module):
    def __init__(self,in_channel,out_channel) -> None:
        super(ResBlock1D,self).__init__()
        self.pre = torch.nn.Identity() if in_channel == out_channel else torch.nn.Conv1d(in_channel,out_channel,1,bias=False)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(out_channel,out_channel,3,1,1,bias=False),
            torch.nn.BatchNorm1d(out_channel),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(out_channel,out_channel,3,1,1,bias=False),
            torch.nn.BatchNorm1d(out_channel),
        )
        self.relu = torch.nn.LeakyReLU()

    def forward(self,x):
        x = self.pre(x)
        out = self.conv(x)
        return self.relu(x+out)

class AtBase(torch.nn.Module):
    def __init__(self,embed_len,n_layers=24,n_heads=4,p_drop=0.1,d_ff=256):
        super(AtBase,self).__init__()
        
        
    def forward(self,intensity,angle):
        intensity = intensity.view(intensity.shape[0],-1,1).type(torch.float)
        angle = angle.view(angle.shape[0],-1,1).type(torch.float)
        
        