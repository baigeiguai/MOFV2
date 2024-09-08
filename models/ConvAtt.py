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

class CAtBlock(torch.nn.Module):
    def __init__(self,in_c,out_c,L,p_dropout):
        super(CAtBlock,self).__init__()
        self.conv = ResBlock1D(in_c,out_c)
        self.pool = torch.nn.AvgPool1d(2,2,L%2)
        self.att = TransformerEncoder((L+1)//2,out_c,1,4,p_dropout,out_c*2)
        
    def forward(self,x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.transpose(-1,-2)
        att_out = self.att(x)
        x = att_out+x 
        x = x.transpose(-1,-2)
        return x 
        


class ConvAtt(torch.nn.Module):
    def __init__(self,p_dropout=0,L=8500):
        super(ConvAtt,self).__init__()
        self.model = torch.nn.Sequential(
            CAtBlock(2,16,L,p_dropout),
            CAtBlock(16,16,L//2,p_dropout),
            CAtBlock(16,32,L//4,p_dropout),
            CAtBlock(32,32,(L+7)//8,p_dropout),
            CAtBlock(32,64,(L+15)//16,p_dropout),
            CAtBlock(64,64,(L+31)//32,p_dropout),
            CAtBlock(64,128,(L+63)//64,p_dropout),
            CAtBlock(128,128,(L+127)//128,p_dropout),
            CAtBlock(128,256,(L+255)//256,p_dropout),
            CAtBlock(256,256,(L+511)//512,p_dropout),
            CAtBlock(256,512,(L+1023)//1024,p_dropout),
            CAtBlock(512,512,(L+2047)//2048,p_dropout),
            CAtBlock(512,1024,(L+4095)//4096,p_dropout),
            CAtBlock(1024,1024,(L+8191)//8192,p_dropout),
        )
        self.cls = torch.nn.Linear(1024,230)
        
    
    def forward(self,intensity,angle):
        intensity = intensity.view(intensity.shape[0],1,-1).type(torch.float)
        angle = angle.view(angle.shape[0],1,-1).type(torch.float)
        x = torch.concat([intensity,angle],dim=1)      
        x = self.model(x)
        x = x.view(x.shape[0],-1)
        return self.cls(x)
        