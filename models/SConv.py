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

ANGLE_NUM = 8500

class SConvBlock(torch.nn.Module):
    def __init__(self,in_c,out_c,pool_k=2,pool_s=2,pool_p=0,pool_type='avg',p_dropout=0.0) -> None:
        super(SConvBlock,self).__init__()
        self.angleConvBlock = ResBlock1D(in_c,out_c)
        self.intensityConvBlock = ResBlock1D(1,1)
        self.pooling = torch.nn.AvgPool1d(pool_k,pool_s,pool_p) if pool_type=='avg' else torch.nn.MaxPool1d(pool_k,pool_s,pool_p)
        if p_dropout>0:
            self.dropout = torch.nn.Dropout(p_dropout)
    
    def forward(self,intensity,angle):
        angle = self.angleConvBlock(angle)
        intensity = self.intensityConvBlock(intensity)
        angle = angle*intensity 
        if hasattr(self,'dropout'):
            angle = self.dropout(angle)
        return self.pooling(intensity),self.pooling(angle)
        
        

class SResTcn(torch.nn.Module):
    def __init__(self,embed_len=32,p_dropout=0.15):
        super(SResTcn,self).__init__()
        self.embed = torch.nn.Embedding(ANGLE_NUM,embed_len)
        self.conv = torch.nn.ModuleList([
            SConvBlock(embed_len,32,2,2,pool_type='max'),
            
            SConvBlock(32,64,2,2,pool_type='max'),
            
            SConvBlock(64,128,2,2,pool_type='max'),
            
            SConvBlock(128,128,2,2,1,pool_type='max'),
            
            SConvBlock(128,256,2,2,pool_type='avg'),
            
            SConvBlock(256,256,2,2,1,pool_type='avg'),
            
            SConvBlock(256,512,2,2,1,pool_type='avg'),
            
            SConvBlock(512,512,2,2,pool_type='avg'),
            
            SConvBlock(512,512,2,2,pool_type='avg',p_dropout=p_dropout),
            
            SConvBlock(512,512,2,2,pool_type='avg',p_dropout=p_dropout),
            
            SConvBlock(512,512,2,2,pool_type='avg',p_dropout=p_dropout),
            
            SConvBlock(512,1024,2,2,pool_type='avg',p_dropout=p_dropout),
            
            SConvBlock(1024,1024,2,2,pool_type='avg'),

            
        ])
        self.linear = torch.nn.Linear(1024,230)
        
    def forward(self,intensity,angle):
        angle = angle.view(angle.shape[0],-1).type(torch.long)
        intensity = intensity.view(intensity.shape[0],-1,1)/100
        x = self.embed(angle)
        x = x.transpose(1,2)
        intensity = intensity.transpose(1,2)
        for conv in self.conv:
            intensity,x = conv(intensity,x)
            
        x = x.view(x.shape[0],-1)
        
        return self.linear(x)
    

if __name__ =='__main__':

    
    device = torch.device('cuda:0')
    # model = ResTcn(2).to(device)
    # inp = torch.randn((16,2,8500)).to(device)
    # out = model(inp)
    # print(out.shape)
    
    
     

    