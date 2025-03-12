import sys

from sympy import limit 
sys.path.append("..")
import json
import os
import argparse
import numpy as np 
from torch import multiprocessing 
from tqdm import tqdm 
import logging
import random 


logging.basicConfig()
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,required=True)
parser.add_argument('--to_path',type=str,required=True)
parser.add_argument('--package_size',type=int,default=50000)
parser.add_argument('--worker_num',type=int,default=15)
parser.add_argument('--split_rate',type=float,default=0.5)

args = parser.parse_args()
dirs = os.listdir(args.to_path)
dest_path = os.path.join(args.to_path,str(max([-1]+[int(i)for i in dirs])+1))
os.makedirs(dest_path)
with open(os.path.join(dest_path,'config.json'),'w') as json_file:
    json_file.write(json.dumps(vars(args)))

train_data = []
test_data = []

for group in range(0,230):
    group_path = os.path.join(args.data_path,str(group))
    sym_in_group = os.listdir(group_path)
    cnt = len(sym_in_group)
    if cnt == 0 :
        continue
    elif cnt == 1 :
        train_data += [os.path.join(group_path,s) for s in sym_in_group]
        test_data += [os.path.join(group_path,s) for s in sym_in_group]
    else:
        train_cnt = int(cnt*args.split_rate)
        train_data += [os.path.join(group_path,s) for s in sym_in_group[:train_cnt]]
        test_data += [os.path.join(group_path,s) for s in sym_in_group[train_cnt:]]
    

logging.info('训练数据共:%d条'%len(train_data))
logging.info('测试数据共:%d条'%len(test_data))

random.shuffle(train_data)
random.shuffle(test_data)

FIXED_1D_LENGTH = 850
MAX_ANGLE = 90
MIN_ANGLE = 5 
def change_data_to_1d(x,y):
    res = [0 for i in range(FIXED_1D_LENGTH)]
    L = len(x)
    for i in range(L):
        idx = int((x[i]-MIN_ANGLE)*(FIXED_1D_LENGTH//(MAX_ANGLE-MIN_ANGLE)))
        res[idx] = max (res[idx],y[i])
    return res 



COORDINATE_NUMBER_LIMIT=500

def process_wrap(cry_paths:list,process_idx:int,stage:str):
    features = []
    labels230 = []
    crystal_name = [] 
    labels7 = []
    lattices_list = []
    atomic_labels_list = []
    cart_coords_list = []
    mask_list = [] 
    for cry_path in cry_paths:
        # print(cry_path)
        try :
            data = np.load(cry_path,allow_pickle=True,encoding='latin1')
        except:
            logging.info("bad file:",cry_path)
            continue
        pattern = data.item().get('pattern')
        features.append(change_data_to_1d(pattern.x,pattern.y))
        labels230.append(data.item().get('space_group'))
        crystal_name.append(data.item().get('crystal_name'))
        labels7.append(data.item().get('crystal_system'))
        atomic_number = data.item().get('atomic_numbers')
        cart_coord = data.item().get('cart_coords')
        lattices_list.append(data.item().get('lattice'))
        L = len(atomic_number)
        if L < COORDINATE_NUMBER_LIMIT:
            atomic_number = list(atomic_number) + [0 for i in range(COORDINATE_NUMBER_LIMIT-L)]
            cart_coord = np.concatenate([cart_coord,np.zeros((COORDINATE_NUMBER_LIMIT-L,3))],axis=0)
        atomic_labels_list.append(atomic_number)
        mask_list.append(np.concatenate([np.ones(L),np.zeros(COORDINATE_NUMBER_LIMIT-L)],axis=0))
        cart_coords_list.append(cart_coord)
            
        
    features = np.array(features)
    labels230 = np.array(labels230)
    crystal_name = np.array(crystal_name)
    labels7 = np.array(labels7)
    lattices_list = np.array(lattices_list)
    atomic_labels_list = np.array(atomic_labels_list)
    mask_list = np.array(mask_list)
    cart_coords_list = np.array(cart_coords_list)
    
    
    save_path = os.path.join(dest_path,"%s_%d"%(stage,process_idx))
    np.save(save_path,{'features':features,
                       "labels7":labels7,
                       'labels230':labels230,
                       'crystal_name':crystal_name,
                       'lattices':lattices_list,
                       'atomic_labels':atomic_labels_list,
                       'mask':mask_list,
                       'cart_coords':cart_coords_list}) # type: ignore
    logging.info('保存路径:%s'%save_path)
    logging.info('保存的样本数量:%d'%len(labels230))
    return None


def star(args):
    process_wrap(**args)

all_args = []
all_args += [dict(cry_paths=train_data[l:l+args.package_size],process_idx = l/args.package_size,stage='train') for l in range(0,len(train_data),args.package_size)]
all_args += [dict(cry_paths=test_data[l:l+args.package_size],process_idx = l/args.package_size,stage='test') for l in range(0,len(test_data),args.package_size)]

if __name__ == '__main__':
    with multiprocessing.Pool(args.worker_num) as pool :
        list(tqdm(pool.imap_unordered(star,all_args),desc='把数据分割打包'))


