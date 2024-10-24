import torch 
from models.ResTcn_8500 import ResTcn
from models.TransformerEncoder import TransformerEncoder


class ResBlock2D(torch.nn.Module):
    def __init__(self,in_channel,out_channel) -> None:
        super(ResBlock2D,self).__init__()
        self.pre = torch.nn.Identity() if in_channel == out_channel else torch.nn.Conv2d(in_channel,out_channel,(1,1),bias=False)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(out_channel,out_channel,(1,3),(1,1),(0,1),bias=False),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(out_channel,out_channel,(1,3),(1,1),(0,1),bias=False),
            torch.nn.BatchNorm2d(out_channel),
        )
        self.relu = torch.nn.LeakyReLU()

    def forward(self,x):
        x = self.pre(x)
        out = self.conv(x)
        return self.relu(x+out)

class PatchConvModule(torch.nn.Module):
    def __init__(self,h=850,w=10):
        super(PatchConvModule,self).__init__()        
        self.h = h 
        self.w = w
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1,2,(1,3),(1,1),(0,1)),
            torch.nn.AvgPool2d((1,2),(1,2)),
            torch.nn.Conv2d(2,4,(1,3),(1,1),(0,1)),
            torch.nn.AvgPool2d((1,2),(1,2),(0,1)),
            torch.nn.Conv2d(4,8,(1,3),(1,1),(0,1)),
            torch.nn.AvgPool2d((1,2),(1,2),(0,1)),
            torch.nn.Conv2d(8,16,(1,3),(1,1),(0,1)),
            torch.nn.AvgPool2d((1,2),(1,2)),
        )
    def forward(self,x):
        x = x.view(x.shape[0],1,self.h,self.w) 
        x =  self.conv(x)
        x = x.view(x.shape[0],-1,self.h)
        return x 

class AttentionModule(torch.nn.Module):
    def __init__(self,patch_len=10,seq_len=8500,embed_len=15,n_layers=8,p_dropout=0.25,d_ff=256):
        super(AttentionModule,self).__init__()
        self.conv_out_c = 16
        self.to_seq_len = seq_len//patch_len
        self.embed  = torch.nn.Embedding(self.to_seq_len,embed_len)
        self.index = torch.nn.Parameter(torch.arange(0,self.to_seq_len).view(1,-1),requires_grad=False)
        self.patch_conv = PatchConvModule(self.to_seq_len,patch_len)
        self.att = TransformerEncoder(self.to_seq_len,self.conv_out_c+1+embed_len,n_layers,p_dropout,d_ff)
    def forward(self,raw_intensity,angle):
        intensity = raw_intensity.view(raw_intensity.shape[0],1,self.to_seq_len,-1)
        intensity_patch_val =  intensity.max(dim=-1).values.view(intensity.shape[0],1,self.to_seq_len)
        att_mask = intensity_patch_val<1e-5
        patch_feature = self.patch_conv(intensity)
        patch_feature = patch_feature.transpose(-1,-2)
        embed_feature = self.embed(self.index.repeat(angle.shape[0],1))
        x = torch.concat([patch_feature,embed_feature,intensity_patch_val.view(intensity_patch_val.shape[0],-1,1)],dim=-1)
        x = self.att(x,att_mask)
        x = x.max(dim=1).values
        return x
        
    
class HopeV1_Con(torch.nn.Module):    
    def __init__(self):
        super(HopeV1_Con,self).__init__()
        self.conv_module = ResTcn(p_dropout=0.25)
        self.att = AttentionModule()
        self.conv_feature_len = 1024
        self.att_feature_len = 32 
        self.cls_sp = torch.nn.Sequential(   
            torch.nn.Linear(self.conv_feature_len+self.att_feature_len,230),
            # torch.nn.Linear(512,230),
        )
        self.cls_cs = torch.nn.Sequential(   
            torch.nn.Linear(self.conv_feature_len+self.att_feature_len,7),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(256,64),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(64,7),
        )
        self.cls_lt = torch.nn.Sequential(   
            torch.nn.Linear(self.conv_feature_len+self.att_feature_len,6),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(256,64),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(64,6),
        )
    def forward(self,intensity,angle):
        conv_feature = self.conv_module(intensity,angle)
        # print("conv_feature_max",torch.max(conv_feature))
        atten_feature = self.att(intensity,angle)
        # print("atten_feature_max",torch.max(atten_feature))
        x = torch.concat([conv_feature,atten_feature],dim=-1)
        # x = self.cls(x)
        # return self.cls_sp(x),self.cls_cs(x),self.cls_lt(x)
        return x 
    
if __name__ == '__main__':
    device = torch.device("cuda:4")
    model = PatchConvModule().to(device) 
    data = torch.rand((8,1,850,10)).to(device)
    out = model(data)
    print(out.shape)