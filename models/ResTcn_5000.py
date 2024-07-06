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
    
class ResTcn(torch.nn.Module):
    def __init__(self):
        super(ResTcn,self).__init__()
        self.conv = torch.nn.Sequential(
            ResBlock1D(2,32),
            torch.nn.AvgPool1d(2,2),
            ResBlock1D(32,32),
            torch.nn.AvgPool1d(2,2),
            
            ResBlock1D(32,64),
            torch.nn.AvgPool1d(2,2),
            ResBlock1D(64,64),
            torch.nn.AvgPool1d(2,2,1),
            
            ResBlock1D(64,128),
            torch.nn.AvgPool1d(2,2,1),
            ResBlock1D(128,128),
            torch.nn.AvgPool1d(2,2,1),
            
            ResBlock1D(128,256),
            torch.nn.AvgPool1d(2,2,1),
            ResBlock1D(256,256),
            torch.nn.AvgPool1d(2,2),
            
            ResBlock1D(256,512),
            torch.nn.AvgPool1d(2,2),
            ResBlock1D(512,512),
            torch.nn.AvgPool1d(2,2),
            
            ResBlock1D(512,1024),
            torch.nn.AvgPool1d(2,2,1),            
            ResBlock1D(1024,1024),
            torch.nn.AvgPool1d(2,2,1),
            
            ResBlock1D(1024,1024),
            torch.nn.AvgPool1d(2,2),
            
            
        )
        self.mlp = torch.nn.Linear(1024,230)
    
    def forward(self,intensity,angle):
        intensity = intensity.view(intensity.shape[0],1,-1)
        angle = angle.view(angle.shape[0],1,-1)
        data = torch.concat([intensity,angle],dim=1)
        embed = self.conv(data)
        # return embed
        embed = embed.view(embed.shape[0],-1)
        return self.mlp(embed)
        
    
if __name__ == '__main__':
    device = torch.device("cuda:4")
    model = ResTcn().to(device)
    intensity = torch.randn(16,1,5000).to(device)
    angle = torch.randn(16,1,5000).to(device)
    out = model(intensity,angle)
    print(out.shape)
    