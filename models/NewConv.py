import torch 

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

class NewConv(torch.nn.Module):
    def __init__(self,patch_len=10,seq_L=850):
        super(NewConv,self).__init__()
        self.patch_len=10
        self.seq_L=850
        self.conv = torch.nn.Sequential(
            ResBlock1D(20,32),
            torch.nn.AvgPool1d(2,2),
            
            ResBlock1D(32,64),
            torch.nn.AvgPool1d(2,2,1),
            
            ResBlock1D(64,128),
            torch.nn.AvgPool1d(2,2,1),
            
            ResBlock1D(128,256),
            torch.nn.AvgPool1d(2,2,1),
            
            ResBlock1D(256,256),
            torch.nn.AvgPool1d(2,2),
            
            ResBlock1D(256,512),
            torch.nn.AvgPool1d(2,2,1),
            
            ResBlock1D(512,512),
            torch.nn.AvgPool1d(2,2,1),
            
            ResBlock1D(512,1024),
            torch.nn.AvgPool1d(2,2),
            
            ResBlock1D(1024,1024),
            torch.nn.AvgPool1d(2,2),
            
            ResBlock1D(1024,1024),
            torch.nn.AvgPool1d(2,2),
        )
        
        self.cls = torch.nn.Linear(1024,230)
        
    def forward(self,intensity,angle):
        intensity = intensity.view(intensity.shape[0],-1,1).type(torch.float)
        angle = angle.view(angle.shape[0],-1,1).type(torch.float)
        x = torch.concat([intensity,angle],dim=-1)
        x = x.view(x.shape[0],self.seq_L,-1)
        x = x.transpose(-1,-2)
        x = self.conv(x)
        x = x.view(x.shape[0],-1)
        x = self.cls(x)
        return x       