from torch import multiprocessing 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch 
import numpy as np


TO_XRD_LENGTH = 8500
ANGLE_START = 5 
ANGLE_END = 90
 
class XrdData(Dataset):
    def __init__(self,file_path):
        data = np.load(file_path,allow_pickle=True,encoding='latin1')
        self.intensity = data.item().get('intensitys')
        self.angle = data.item().get('angles')
        self.labels230 = data.item().get('labels230')
        self.labels7 = data.item().get('labels7')
        self.lattice = data.item().get('lattices')
        self.atomic_labels = data.item().get('atomic_labels')
        self.mask = data.item().get('mask')
        self.cart_coords = data.item().get('cart_coords')
        if self.intensity is None:
            self.intensity = data.item().get('features')
        if self.angle is None :
            self.angle = np.arange(ANGLE_START,ANGLE_END,(ANGLE_END-ANGLE_START)/TO_XRD_LENGTH).reshape(1,-1).repeat(len(self.labels230),axis=0).astype(np.float32)

        self.index = np.arange(0,TO_XRD_LENGTH).reshape(1,-1).repeat(len(self.labels230),axis=0)
        # atomic_number  = data.item().get('atomic_number')
        # print(self.features.shape,self.angle.shape,self.labels230.shape,self.labels7.shape)
        
    def __getitem__(self, index) :
        # return [self.features[index],self.angle,self.labels230[index],self.labels7[index]]
        # torch.Size([16, 850]) torch.Size([16, 850]) torch.Size([16]) torch.Size([16, 3, 3]) torch.Size([16, 500]) torch.Size([16, 500]) torch.Size([16, 500, 3])
        return [self.intensity[index],
                self.angle[index],
                self.labels230[index],
                self.index[index]]
                # self.lattice[index],
                # self.atomic_labels[index],
                # self.mask[index],
                # self.cart_coords[index]]
        
    def __len__(self):
        return len(self.labels230)
    
if __name__ == '__main__':
    t = XrdData('/home/ylh/code/MyExps/MOFV2/data/Pymatgen_Wrapped_Short/0/train_0.npy')
    dataloader = DataLoader(t,16,True)
    for data in dataloader :
        [a,b,c] = data 
        print(a.shape,b.shape,c.shape)
        # [a,b,c,d,e,f,g] = data 
        # print(a.shape,b.shape,c.shape,d.shape,e.shape,f.shape,g.shape)
        # print(torch.concat([e.view(e.shape[0],-1,1),g],dim=-1).shape)
        # torch.Size([16, 850]) torch.Size([16, 850]) torch.Size([16]) torch.Size([16, 3, 3]) torch.Size([16, 500]) torch.Size([16, 500]) torch.Size([16, 500, 3])
        break
    