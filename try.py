import torch 
from torchinfo import summary
# from models.BiMamba import BiMamba,MambaConfig

# device = torch.device('cuda:3')
# inp = torch.randn((16,8500,16)).to(device)
# config = MambaConfig(d_model=16,n_layers=2,d_state=16)
# model = BiMamba(config).to(device)
# out = model(inp)
# print(out.shape)


# from models.ExplorerV1 import ExplorerV1
# device = torch.device('cuda:3')
# angle = torch.randn((16,8500)).to(device)
# intensity = torch.randn((16,8500)).to(device)
# model = ExplorerV1(16,4).to(device)
# out = model(intensity,angle)
# print(out.shape)
# from utils.logger import Log
# log = Log(__name__)
# logger = log.get_log()
# logger.info('hello')

# from models.ExplorerV1 import ExplorerV1
# from torchsummary import summary
# device = torch.device('cuda:0')
# model = ExplorerV1(32,1).to(device)
# summary(model,input_size=[(1,8500),(1,8500)])
# intensity = torch.randn((256,1,8500)).to(device)    
# angle = torch.randn((256,1,8500)).to(device)
# out1,out2 = model(intensity,angle)
# print(out1.shape,out2.shape)

# from mamba_ssm import Mamba
# from torchinfo import summary
# batch, length, dim = 16, 8500, 2
# device = torch.device("cuda:1")
# # torch.cuda.set_device(1)
# x = torch.randn((batch, length, dim)).to(device)
# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=dim, # Model dimension d_model
#     d_state=32,  # SSM state expansion factor, typically 64 or 128
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to(device)
# summary(model,(batch,length,dim))

# from models.ExplorerV2 import ExplorerV2
# batch,length = 16,8500
# device = torch.device("cuda:2")
# angle = torch.randn(batch,length).to(device)
# intensity = torch.randn(batch,length).to(device)
# model = ExplorerV2(2,32,32,0).to(device)
# out = model(angle,intensity)
# print(out.shape)

# summary(model,[(batch,length),(batch,length)])




# from models.RetryViT1D import RetryViT
# batch,length = 16,8500
# device = torch.device("cuda:3")
# angle = torch.randn(batch,length).to(device)
# intensity = torch.randn(batch,length).to(device)
# model = RetryViT().to(device)
# out = model(intensity,angle)
# print(out.shape)

# from models.XAt_base import XrdAttentionBase
# from utils.dataset import XrdData
# from torch.utils.data import DataLoader

# device = torch.device("cuda:2")
# file = '/home/ylh/code/MyExps/MOFV2/data/Pymatgen_Wrapped/0/test_0.npy'
# model = XrdAttentionBase().to(device)
# xrd_dataset = XrdData(file)
# dataloader = DataLoader(xrd_dataset,32,num_workers=20)
# for data in dataloader:
#     intensity,angle,labels230 = data[0].type(torch.float).to(device),data[1].type(torch.float).to(device),data[2].to(device)
#     out = model(intensity,angle)
#     break
# batch,length = 32,8500
# angle = torch.arange(0,length).view(1,-1).repeat((batch,1)).to(device)
# intensity = torch.randn(batch,length).to(device)
# out = model(angle,intensity)
# summary(model,[(batch,length),(batch,length)])

# from models.RawEmbedConv import RawEmbedConv
# device = torch.device("cuda:3")
# model = RawEmbedConv().to(device)
# batch,length = 32,8500
# angle = torch.arange(0,length).view(1,-1).repeat((batch,1)).to(device)
# intensity = torch.randn(batch,length).to(device)
# a = model(intensity,angle)

# summary(model,[(batch,length),(batch,length)])

# print(a.shape)


# from models.ConcatEmbedConv import ConcatEmbedConv
# device = torch.device("cuda:1")
# model = ConcatEmbedConv().to(device)
# batch,length = 32,8500
# angle = torch.arange(0,length).view(1,-1).repeat((batch,1)).to(device)
# intensity = torch.randn(batch,length).to(device)
# a = model(intensity,angle)

# summary(model,[(batch,length),(batch,length)])

# print(a.shape)

from models.AtBase import AtBase
device = torch.device("cuda:1")
model = AtBase(n_layers=24).to(device)
batch,length = 32,850
angle = torch.arange(0,length).view(1,-1).repeat((batch,1)).to(device)
intensity = torch.randn(batch,length).to(device)
a = model(intensity,angle)
summary(model,[(batch,length),(batch,length)])
print(a.shape)