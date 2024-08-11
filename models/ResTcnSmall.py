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
    def __init__(self,in_c=2,p_dropout=0.1):
        super(ResTcn,self).__init__()
        self.in_c = in_c
        
        self.conv = torch.nn.Sequential(
            
            ResBlock1D(in_c,8),
            torch.nn.Conv1d(8,8,3,2,1),
            
            ResBlock1D(8,8),
            torch.nn.Conv1d(8,8,3,2,1),
            
            ResBlock1D(8,16),
            torch.nn.Conv1d(16,16,3,2,1),
            
            ResBlock1D(16,16),
            torch.nn.Conv1d(16,16,3,2,1),
            
            ResBlock1D(16,32),
            torch.nn.Conv1d(32,32,3,2,1),

            ResBlock1D(32,32),
            torch.nn.Conv1d(32,32,3,2,1),

            ResBlock1D(32,64),
            torch.nn.Conv1d(64,64,3,2,1),
            
            ResBlock1D(64,128),
            torch.nn.Conv1d(128,128,3,2,1),

            ResBlock1D(128,128),
            torch.nn.Conv1d(128,128,3,2,1),

            ResBlock1D(128,256),
            torch.nn.Conv1d(256,256,3,2,1),

            ResBlock1D(256,256),
            torch.nn.Conv1d(256,256,3,2,1),

            ResBlock1D(256,512),
            torch.nn.Conv1d(512,512,3,2,1),
            torch.nn.Dropout(p_dropout),

            ResBlock1D(512,512),
            torch.nn.Conv1d(512,512,3,2,1),
            torch.nn.Dropout(p_dropout),

            ResBlock1D(512,1024),
            torch.nn.Conv1d(1024,1024,3,2,1),
            torch.nn.Dropout(p_dropout),


            torch.nn.Flatten(),
            torch.nn.Linear(1024,230),                

        )
        
    def forward(self,x):
        x = x.view(x.shape[0],self.in_c,-1)
        return self.conv(x)


if __name__ =='__main__':
    from torchinfo import summaryw
    device = torch.device('cuda:3')
    model = ResTcn(2,0.5).to(device)
    inp = torch.randn((16,2,8500)).to(device)
    out = model(inp)
    summary(model,(16,2,8500))
    print("out.shape:",out.shape)
    
    
    
     

    