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

# from models.AtBase import AtBase
# device = torch.device("cuda:0")
# model = AtBase(n_layers=24,n_heads=1).to(device)
# batch,length = 32,850
# angle = torch.arange(0,length).view(1,-1).repeat((batch,1)).to(device)
# intensity = torch.randn(batch,length).to(device)
# a = model(intensity,angle)
# summary(model,[(batch,length),(batch,length)])
# print(a.shape)

# from models.AtLBase import AtLBase
# device = torch.device("cuda:1")
# model = AtLBase(n_layers=8,embed_len=512,d_ff=1024).to(device)
# batch,length = 256,850
# angle = torch.arange(0,length).view(1,-1).repeat((batch,1)).to(device)
# intensity = torch.randn(batch,length).to(device)
# a,_,_ = model(intensity,angle)
# summary(model,[(batch,length),(batch,length)])
# print(a.shape)

# from pymatgen.core import Structure
# from pymatgen.analysis.diffraction.xrd import XRDCalculator
# import argparse  

# ANGLE_START = 5
# ANGLE_END = 90

# space_group_map_dict = {}
# for i in range(1, 3):
#     space_group_map_dict[i] = 1
# for i in range(3, 16):
#     space_group_map_dict[i] = 2
# for i in range(16, 75):
#     space_group_map_dict[i] = 3
# for i in range(75, 143):
#     space_group_map_dict[i] = 4
# for i in range(143, 168):
#     space_group_map_dict[i] = 5
# for i in range(168, 195):
#     space_group_map_dict[i] = 6
# for i in range(195, 231):
#     space_group_map_dict[i] = 7

# parser = argparse.ArgumentParser()
# parser.add_argument('--data_path',type=str,required=True)
# args = parser.parse_args()

# structure  = Structure.from_file(args.data_path,primitive=True)
# space_group = structure.get_space_group_info()[1]-1
# print("space group:%d"%space_group)
# crystal_system = space_group_map_dict[space_group+1]-1
# print("crystal_system:%d"%crystal_system)
# atomic_numbers = structure.atomic_numbers
# print("len:",len(atomic_numbers),"atomic_numbers:",atomic_numbers[:500])
# cart_coords = structure.cart_coords
# print("len:",len(cart_coords),"cart_coords:",cart_coords[:500])
# frac_coords = structure.frac_coords
# print("len:",len(frac_coords),"frac_coords:",frac_coords)
# lattice = structure.lattice
# print("lattice:",lattice.metric_tensor)


# device = torch.device("cuda:6")
# from models.ResTcn_8500 import ResTcn
# model = ResTcn(2,0).to(device)
# batch,length = 256,8500
# angle = torch.arange(0,length).view(1,-1).repeat((batch,1)).to(device)
# intensity = torch.randn(batch,length).to(device)
# a = model(intensity,angle)
# summary(model,[(batch,length),(batch,length)])
# print(a.shape)

# import argparse 

# parser = argparse.ArgumentParser()
# parser.add_argument('--model_path',type=str)
# args = parser.parse_args()

# device = torch.device("cuda:6")
# model = torch.load(args.model_path,map_location=device).to(device)
# model.TCN.__delitem__(30)
# batch,length = 32,8500
# angle = torch.arange(0,length).view(1,-1).repeat((batch,1)).to(device)
# intensity = torch.randn(batch,length).to(device)
# a = model(intensity,angle)
# # summary(model,[(batch,length),(batch,length)])
# print(a.shape)

# device = torch.device("cuda:6")
# from models.AtLBase import AtLBase 

# model = AtLBase(embed_len=8,n_layers=4,p_drop=0).to(device)
# batch,length = 8,8500
# index = torch.arange(0,length).view(1,-1).repeat((batch,1)).to(device)
# intensity = torch.randn(batch,length).to(device)
# a,b = model(intensity,index)
# summary(model,[(batch,length),(batch,length)])
# print(a.shape,b.shape)

# device = torch.device("cuda:6")
# from models.AttDistil import AttDistil 
# model = AttDistil().to(device)
# batch,length = 32,8500
# index = torch.arange(0,length).view(1,-1).repeat((batch,1)).to(device)
# intensity = torch.randn(batch,length).to(device)
# a,b = model(intensity,index)
# summary(model,[(batch,length),(batch,length)])
# # print(a.shape)
# print(a.shape,b.shape)


# device = torch.device("cuda:6")
# from models.ConvAtt import ConvAtt
# model = ConvAtt().to(device)
# batch,length = 32,8500
# angle = torch.arange(0,length).view(1,-1).repeat((batch,1)).to(device)
# intensity = torch.randn(batch,length).to(device)
# a = model(intensity,angle)
# summary(model,[(batch,length),(batch,length)])
# print(a.shape)

# device = torch.device("cuda:0")
# from models.NewConv import NewConv
# model = NewConv().to(device)
# batch,length = 8,8500
# angle = torch.arange(0,length).view(1,-1).repeat((batch,1)).to(device)
# intensity = torch.randn(batch,length).to(device)
# a = model(intensity,angle)
# summary(model,[(batch,length),(batch,length)])
# print(a.shape)

# device = torch.device("cuda:6")
# from models.RetryViT1D import RetryViT
# model = RetryViT().to(device)
# batch,length = 8,8500
# angle = torch.arange(0,length).view(1,-1).repeat((batch,1)).to(device)
# intensity = torch.randn(batch,length).to(device)
# a = model(intensity,angle)
# summary(model,[(batch,length),(batch,length)])
# print(a.shape)

# device = torch.device("cuda:6")
# from models.ConvAttV2 import ConvAttV2 
# model = ConvAttV2().to(device)
# batch,length = 8,8500
# angle = torch.arange(0,length).view(1,-1).repeat((batch,1)).to(device)
# intensity = torch.randn(batch,length).to(device)
# a = model(intensity,angle)
# summary(model,[(batch,length),(batch,length)])
# print(a.shape)

# device = torch.device("cuda:0")
# from models.AtLV2 import AtLV2
# model = AtLV2().to(device)
# batch,length = 4,8500
# angle = torch.arange(0,length).view(1,-1).repeat((batch,1)).to(device)
# angle = angle.type(torch.float)/100 + 5 
# intensity = torch.randn(batch,length).to(device)
# a = model(intensity,angle)
# summary(model,[(batch,length),(batch,length)])
# print(a.shape)

device = torch.device("cuda:6")
from models.AtLSmall import AtLSmall
model = AtLSmall().to(device)
batch,length,zero_pad_len = 64,8500,22
angle = torch.arange(0,length-zero_pad_len).view(1,-1)
angle = angle.type(torch.float)/100 + 5 
zeros = torch.zeros(zero_pad_len).view(1,-1)
angle = torch.concat([angle,zeros],dim=-1).repeat((batch,1)).to(device)
intensity = torch.randn(batch,length).to(device)
a = model(intensity,angle)
summary(model,[(batch,length),(batch,length)])
print(a.shape)