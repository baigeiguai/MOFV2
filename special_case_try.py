import os 
import torch 
import argparse
import numpy as np 
from utils.dataset import sp2lt,sp2cs,sp2pg


parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,required=True)
parser.add_argument('--model_path',type=str,required=True)

args = parser.parse_args()
tasks = ['sp','lt','cs','pg']
models = torch.load(args.model_path,map_location=torch.device('cpu'))
print(models)
FIXED_1D_LENGTH = 8500
MAX_ANGLE = 90
MIN_ANGLE = 5 
def change_data_to_1d(x,y):
    res = [0 for i in range(FIXED_1D_LENGTH)]
    L = len(x)
    for i in range(L):
        idx = int((x[i]-MIN_ANGLE)*(FIXED_1D_LENGTH//(MAX_ANGLE-MIN_ANGLE)))
        res[idx] = max (res[idx],y[i])
    return res 

res = []
not_ids = [75-1,79-1,81-1,82-1] 
for file in os.listdir(args.data_path):
    file_path = os.path.join(args.data_path,file)
    data = np.load(file_path,allow_pickle=True,encoding='latin1')
    pattern = data.item().get('pattern')
    intensity = change_data_to_1d(pattern.x,pattern.y)
    labels230 = data.item().get('space_group')
    lattice_type = sp2lt[labels230]
    point_group = sp2pg[labels230]
    crystal_system = sp2cs[labels230]
    crystal_name = data.item().get('crystal_name')
    angle = np.arange(MIN_ANGLE,MAX_ANGLE,(MAX_ANGLE-MIN_ANGLE)/FIXED_1D_LENGTH).astype(np.float64)
    intensity,angle = torch.tensor(intensity).view(1,-1).type(torch.float),torch.tensor(angle).view(1,-1).type(torch.float)
    labels = {
        'sp':labels230,
        'lt':lattice_type,
        'cs':crystal_system,
        'pg':point_group,
    }
    features = models['rep'](intensity,angle)

    for t in tasks:
        out = models[t](features)
        out = out.softmax(dim=-1)
        vals,idxs = out.topk(k=5,dim=1,largest=True,sorted=True)
        if vals[0] == 74 :
            res.append(file_path)
            
print(res)
        # print('task:%s,label:%s,predicted:%s'%(
        #     t,
        #     labels[t],
        #     idxs,
        # ))
        # if t=='sp':
        #     print(vals)

# out = encoder(input_data,idx).softmax(dim=-1)
# # torch.save(encoder.state_dict(),'../XRDMamba/static/try.pt')
# print(out.shape)
# vals,idxs = out.topk(k=5,dim=1,largest=True,sorted=True)
# print(idxs)
# print(vals)

