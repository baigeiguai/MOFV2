from pymatgen.core import Structure # ,Lattice, Molecule
from pymatgen.analysis.diffraction.xrd import XRDCalculator
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
ANGLE_START = 5
ANGLE_END = 90
def change_data_to_1d(x,y):
    res = [0 for i in range(FIXED_1D_LENGTH)]
    L = len(x)
    for i in range(L):
        idx = int((x[i]-MIN_ANGLE)*(FIXED_1D_LENGTH//(MAX_ANGLE-MIN_ANGLE)))
        res[idx] = max (res[idx],y[i])
    return res 
space_group_map_dict = {}
for i in range(1, 3):
    space_group_map_dict[i] = 1
for i in range(3, 16):
    space_group_map_dict[i] = 2
for i in range(16, 75):
    space_group_map_dict[i] = 3
for i in range(75, 143):
    space_group_map_dict[i] = 4
for i in range(143, 168):
    space_group_map_dict[i] = 5
for i in range(168, 195):
    space_group_map_dict[i] = 6
for i in range(195, 231):
    space_group_map_dict[i] = 7
try:
    structure  = Structure.from_file(args.data_path,primitive=True)
    space_group = structure.get_space_group_info()[1]-1
    assert (space_group+1) in space_group_map_dict
    crystal_system = space_group_map_dict[space_group+1]-1
    # cart_coords = structure.cart_coords[:COORDINATE_NUMBER_LIMIT]
    # atomic_numbers = structure.atomic_numbers[:COORDINATE_NUMBER_LIMIT]
    lattice =  structure.lattice.metric_tensor
    xrd = XRDCalculator(wavelength=1.54056,symprec=0.1) # type: ignore 
    pattern = xrd.get_pattern(structure,scaled=True,two_theta_range=(ANGLE_START,ANGLE_END))
except:
    # logging.error('file_path:%s\n'%file_path)
    print('file_path:%s\n'%args.data_path)
# data = np.load(args.data_path,allow_pickle=True,encoding='latin1')
# pattern = data.item().get('pattern')
intensity = change_data_to_1d(pattern.x,pattern.y)
# labels230 = data.item().get('space_group')
labels230 = space_group
lattice_type = sp2lt[labels230]
point_group = sp2pg[labels230]
crystal_system = sp2cs[labels230]
# crystal_name = data.item().get('crystal_name')
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
    print('task:%s,label:%s,predicted:%s'%(
        t,
        labels[t],
        idxs,
    ))
    if t=='sp':
        print(vals)

# out = encoder(input_data,idx).softmax(dim=-1)
# # torch.save(encoder.state_dict(),'../XRDMamba/static/try.pt')
# print(out.shape)
# vals,idxs = out.topk(k=5,dim=1,largest=True,sorted=True)
# print(idxs)
# print(vals)

