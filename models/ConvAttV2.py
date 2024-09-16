import torch 
from models.TransformerEncoder import TransformerEncoder

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

class ConvAttV2(torch.nn.Module):
    def __init__(self,seq_len=532,dim=64,n_layes=48,n_heads=4,p_dropout=0,d_ff=256):
        super(ConvAttV2,self).__init__()
        self.conv = torch.nn.Sequential(
            ResBlock1D(2,32),
            torch.nn.AvgPool1d(2,2),
            
            ResBlock1D(32,32),
            torch.nn.AvgPool1d(2,2),
            
            ResBlock1D(32,64),
            torch.nn.AvgPool1d(2,2,1),
            
            ResBlock1D(64,64),
            torch.nn.AvgPool1d(2,2,1),
        )
        self.att = TransformerEncoder(seq_len=seq_len,dimension=dim,n_layers=n_layes,n_heads=n_heads,p_drop=p_dropout,d_ff=d_ff)
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(64,128),   
            torch.nn.Linear(128,230),
        )
        
    def forward(self,intensity,angle):
        intensity = intensity.view(intensity.shape[0],1,-1).type(torch.float)
        angle = angle.view(angle.shape[0],1,-1).type(torch.float)
        x = torch.concat([intensity,angle],dim=1)
        x = self.conv(x)
        x = x.transpose(-1,-2)
        x = self.att(x)
        x = x.mean(dim=1)
        x =  self.cls(x)
        return x 