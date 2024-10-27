from torch import multiprocessing 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch 
import numpy as np
import os 


TO_XRD_LENGTH = 8500
ANGLE_START = 5 
ANGLE_END = 90
 
sp2lt = {13: 0, 8: 1, 147: 2, 0: 0, 1: 0, 143: 0, 61: 0, 18: 0, 166: 2, 14: 1, 165: 2, 146: 0, 4: 1, 154: 2, 224: 3, 208: 3, 138: 4, 122: 0, 78: 4, 160: 2, 17: 0, 3: 0, 5: 0, 60: 0, 210: 4, 145: 2, 159: 2, 7: 1, 33: 0, 155: 0, 142: 0, 25: 0, 216: 4, 196: 4, 81: 4, 177: 0, 19: 1, 168: 0, 11: 1, 125: 0, 74: 0, 84: 0, 228: 4, 23: 4, 80: 0, 15: 0, 214: 0, 220: 0, 151: 0, 28: 0, 42: 3, 86: 4, 96: 4, 71: 4, 112: 0, 12: 0, 9: 0, 70: 4, 200: 0, 47: 0, 68: 3, 64: 1, 20: 1, 174: 0, 136: 0, 93: 0, 55: 0, 39: 5, 49: 0, 213: 4, 119: 4, 173: 0, 171: 0, 52: 0, 16: 0, 186: 0, 48: 0, 85: 0, 153: 0, 126: 0, 89: 0, 115: 0, 207: 0, 137: 0, 22: 4, 123: 0, 229: 4, 141: 4, 194: 0, 199: 0, 181: 0, 117: 0, 161: 0, 190: 0, 2: 0, 176: 0, 218: 3, 225: 3, 29: 0, 149: 0, 77: 0, 98: 0, 27: 0, 69: 3, 180: 0, 90: 0, 124: 0, 53: 0, 140: 4, 87: 4, 97: 4, 156: 0, 6: 0, 41: 3, 54: 0, 204: 0, 75: 0, 184: 0, 189: 0, 95: 0, 226: 3, 209: 3, 56: 0, 58: 0, 50: 0, 167: 0, 211: 0, 76: 0, 30: 0, 32: 0, 101: 0, 206: 0, 170: 0, 157: 0, 104: 0, 113: 0, 118: 4, 183: 0, 182: 0, 158: 0, 185: 0, 88: 0, 179: 0, 187: 0, 82: 0, 129: 0, 175: 0, 65: 1, 192: 0, 37: 5, 193: 0, 172: 0, 127: 0, 108: 4, 150: 0, 169: 0, 102: 0, 162: 0, 164: 0, 152: 0, 144: 0, 128: 0, 223: 0, 105: 0, 109: 4, 163: 0, 178: 0, 73: 4, 188: 0, 197: 0, 222: 0, 21: 3, 10: 0, 198: 4, 46: 0, 31: 0, 227: 3, 121: 4, 120: 4, 103: 0, 72: 4, 35: 1, 131: 0, 221: 0, 100: 0, 191: 0, 114: 0, 203: 4, 201: 3, 111: 0, 26: 0, 83: 0, 135: 0, 34: 1, 215: 3, 59: 0, 217: 0, 148: 0, 106: 4, 45: 4, 130: 0, 139: 4, 195: 3, 134: 0, 99: 0, 44: 4, 91: 0, 107: 4, 24: 0, 57: 0, 202: 3, 36: 1, 212: 0, 132: 0, 43: 4, 133: 0, 92: 0, 110: 0, 94: 0, 219: 4, 62: 1, 51: 0, 116: 0, 205: 4, 79: 4}

 
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
                self.index[index],
                self.labels7[index],
                sp2lt[self.labels230[index]],
                ]
                # self.lattice[index],
                # self.atomic_labels[index],
                # self.mask[index],
                # self.cart_coords[index]]
        
    def __len__(self):
        return len(self.labels230)
    
    
class ClassedXrdDataset(Dataset):
    def __init__(self,files_dir:str):
        cls_cnt = [0 for i in range(230)]
        labels230 = []
        features = []
        for sp_idx in range(230):
            file_path = os.path.join(files_dir,str(sp_idx),"%d.npy"%sp_idx)
            if not os.path.exists(file_path):
                continue
            data = np.load(file_path)
            # print(data.shape)
            cls_cnt[sp_idx]=data.shape[0]
            if sp_idx == 0 :
                features = data
            else:
                features = np.append(features,data,axis=0)
            labels230 += [sp_idx for i in range(data.shape[0])]
        self.features = features
        self.labels230 = labels230 
        self.angle = np.arange(ANGLE_START,ANGLE_END,(ANGLE_END-ANGLE_START)/TO_XRD_LENGTH).astype(np.float32)
        self.cls_cnt = cls_cnt
        self.new_labels230 = []
        # np.save('/data/ylh/MyExps/MOFV2/data/All_Wrapped/train_features.npy',self.features)
        # np.save('/data/ylh/MyExps/MOFV2/data/All_Wrapped/train_labels230.npy',self.labels230)
    def __getitem__(self, index) :
        return [
            self.features[index],
            self.angle,
            self.labels230[index],
            self.new_labels230[index] if self.new_labels230!=[] else -1,
        ]
    
    def __len__(self):
        return len(self.labels230)

class AllXrdDataset(Dataset) :
    def __init__(self,dir_path):
        self.features = np.load(os.path.join(dir_path,'train_features.npy'),allow_pickle=True)
        self.labels230 = np.load(os.path.join(dir_path,'train_labels230.npy'),allow_pickle=True)
        self.cls_cnt = np.load(os.path.join(dir_path,'train_cls_cnt.npy'),allow_pickle=True)
        self.angle = np.arange(ANGLE_START,ANGLE_END,(ANGLE_END-ANGLE_START)/TO_XRD_LENGTH).astype(np.float32)
        self.new_labels230 = []
    
    def __getitem__(self, index):
        
        return [
            self.features[index],
            self.angle,
            self.labels230[index],
            self.new_labels230[index] if self.new_labels230!=[] else -1,
        ]
    
    def __len__(self):
        return len(self.labels230)
                 


if __name__ == '__main__':
    dataset = AllXrdDataset('/data/ylh/MyExps/MOFV2/data/All_Wrapped')
    print(dataset.__len__())
    print(dataset.cls_cnt)
    dataloader = DataLoader(dataset,8,shuffle=True,num_workers=30,pin_memory=True)
    for [a,b,c,d] in dataloader:
        print(a.shape,b.shape,c.shape,d.shape)
        break