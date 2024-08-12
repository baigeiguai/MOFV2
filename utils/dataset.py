from torch import multiprocessing 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


TO_XRD_LENGTH = 850
ANGLE_START = 5 
ANGLE_END = 90
 
class XrdData(Dataset):
    def __init__(self,file_path):
        data = np.load(file_path,allow_pickle=True,encoding='latin1')
        self.intensity = data.item().get('intensitys')
        self.angle = data.item().get('angles')
        self.labels230 = data.item().get('labels230')
        self.labels7 = data.item().get('labels7')
        if self.intensity is None:
            self.intensity = data.item().get('features')
        if self.angle is None :
            # self.angle = [np.arange(ANGLE_START,ANGLE_END,(ANGLE_END-ANGLE_START)/TO_XRD_LENGTH) for i in range(len(self.labels230)) ]
            self.angle = np.arange(0,TO_XRD_LENGTH).reshape(1,-1).repeat(len(self.labels230),axis=0)
        # atomic_number  = data.item().get('atomic_number')
        # print(self.features.shape,self.angle.shape,self.labels230.shape,self.labels7.shape)
        
    def __getitem__(self, index) :
        # return [self.features[index],self.angle,self.labels230[index],self.labels7[index]]
        return [self.intensity[index],self.angle[index],self.labels230[index]]
        
    def __len__(self):
        return len(self.labels230)
    
if __name__ == '__main__':
    t = XrdData('/home/ylh/code/MyExps/MOFV2/data/Pymatgen_Wrapped/0/test_0.npy')
    dataloader = DataLoader(t,16,True)
    for data in dataloader :
        [a,b,c] = data 
        print(a.shape,b.shape,c.shape)
        print(b[0][50:100])
        break
    