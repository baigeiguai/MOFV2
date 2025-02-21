import torch
import numpy  as np 
from torch.utils.data import DataLoader
from utils.dataset import XrdData
import argparse
from matplotlib import pyplot as plt 
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--encoder",type=str,required=True)
parser.add_argument("--data_path",type=str,required=True)
parser.add_argument("--device",type=int,default=0)
parser.add_argument('--batch_size',type=int,default=5120)
parser.add_argument("--num_workers",type=int,default=20)

args = parser.parse_args()

device = torch.device("cuda:%s"%args.device if torch.cuda.is_available() else 'cpu')
models = torch.load(args.encoder,map_location=device)

xrd_dataset = XrdData(args.data_path)
dataloader = DataLoader(xrd_dataset,args.batch_size,num_workers=args.num_workers)

xi,xa,y = None,None,None
for data in dataloader:
    intensitys , angles,labels230s = data[0].type(torch.float).to(device),data[1].type(torch.float).to(device),data[2].to(device)
    for i in range(labels230s.shape[0]):
        if labels230s[i] == 195 :
            for t in data :
                print(t[i])
            xi = intensitys[i].view(1,-1)
            xa = angles[i].view(1,-1)
            y = [labels230s[i]]
            break
    if xi is not None:
        break

out,atw = models['rep'].att(xi,xa)
atw = atw.view(atw.shape[1],-1)
# atw = atw.sum(dim=0)

inten = xi[0].detach().cpu().numpy()
atw = atw.detach().cpu().numpy()

np.save('atten_case.npy',{
    "xi":inten,
    "at":atw,
})
