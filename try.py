
import torch 
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

from models.ExplorerV1 import ExplorerV1
device = torch.device('cuda:2')
intensity = torch.randn((256,1,8500)).to(device)
angle = torch.randn((256,1,8500)).to(device)
model = ExplorerV1(2,2).to(device)
out1,out2 = model(intensity,angle)
print(out1.shape,out2.shape)




