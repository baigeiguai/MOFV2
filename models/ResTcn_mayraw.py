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


class RES_TCN(torch.nn.Module):
    def __init__(self,in_c=2,p_dropout=0.1):
        super(RES_TCN,self).__init__()
        self.in_c = in_c
        
        self.TCN = torch.nn.Sequential(
            ResBlock1D(in_c,32),
            torch.nn.MaxPool1d(2,2),

            ResBlock1D(32,32),
            torch.nn.MaxPool1d(2,2),

            ResBlock1D(32,64),
            torch.nn.MaxPool1d(2,2),
            
            ResBlock1D(64,128),
            torch.nn.MaxPool1d(2,2),

            ResBlock1D(128,128),
            torch.nn.AvgPool1d(2,2,1),

            ResBlock1D(128,256),
            torch.nn.AvgPool1d(2,2),

            ResBlock1D(256,256),
            torch.nn.AvgPool1d(2,2,1),

            ResBlock1D(256,512),
            torch.nn.AvgPool1d(2,2,1),

            ResBlock1D(512,512),
            torch.nn.AvgPool1d(2,2),

            ResBlock1D(512,1024),
            torch.nn.AvgPool1d(2,2),
            torch.nn.Dropout(p_dropout),

            ResBlock1D(1024,1024),
            torch.nn.AvgPool1d(2,2),
            torch.nn.Dropout(p_dropout),

            ResBlock1D(1024,1024),
            torch.nn.AvgPool1d(2,2),
            torch.nn.Dropout(p_dropout),

            ResBlock1D(1024,1024),
            torch.nn.AvgPool1d(2,2),

            torch.nn.Flatten(),
            torch.nn.Linear(1024,230),                

        )
        
    def forward(self,inp,y=None):
        inp = inp.view(inp.shape[0],self.in_c,-1)
        return self.TCN(inp)


if __name__ =='__main__':
    model = RES_TCN(2)

    # total_params = sum(p.numel() for p in model.parameters())
    # total_params += sum(p.numel() for p in model.buffers())
    # print(total_params/1024/1024)
    inp = torch.randn((4,2,8500)).type(torch.float)
    print(inp.shape)
    inp = model(inp)
    print(inp.shape)
     

    