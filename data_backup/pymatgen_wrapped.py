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
    # elif cnt == 1 :
    #     train_data += [os.path.join(group_path,s) for s in sym_in_group]
    #     test_data += [os.path.join(group_path,s) for s in sym_in_group]
    else:
        train_cnt = int(cnt*args.split_rate)
        
        train_data += [os.path.join(group_path,s) for s in sym_in_group[:train_cnt]]
        test_data += [os.path.join(group_path,s) for s in sym_in_group[train_cnt:]]
    

logging.info('训练数据共:%d条'%len(train_data))
logging.info('测试数据共:%d条'%len(test_data))

random.shuffle(train_data)
random.shuffle(test_data)

print("train_cnt:",len(train_data))

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


# PADDING_FRAC_COORDS_LENGTH = 65536


def process_wrap(cry_paths:list,process_idx:int,stage:str):
    features = []
    labels230 = []
    atomic_number = []
    crystal_name = [] 
    labels7 = []
    # frac_coords_list = []
    # atom_labels_list = [] 
    for cry_path in cry_paths:
        # print(cry_path)
        data = np.load(cry_path,allow_pickle=True,encoding='latin1')
        # frac_coords = data.item().get('frac_coords')
        # atom_labels = data.item().get('atom_labels')
        # if frac_coords.shape[0] != len(atom_labels) or frac_coords.shape[0] > PADDING_FRAC_COORDS_LENGTH:
        #     continue
        pattern = data.item().get('pattern')
        features.append(change_data_to_1d(pattern.x,pattern.y))
        labels230.append(data.item().get('space_group'))
        atomic_number.append(data.item().get('atomic_number'))
        crystal_name.append(data.item().get('crystal_name'))
        labels7.append(data.item().get('crystal_system'))
        # frac_coords_list.append(np.concatenate((frac_coords,np.zeros([PADDING_FRAC_COORDS_LENGTH-frac_coords.shape[0],3]))))
        # atom_labels_list.append(list(atom_labels)+[0 for i in range(PADDING_FRAC_COORDS_LENGTH-len(atom_labels))])
    features = np.array(features)
    labels230 = np.array(labels230)
    atomic_number = np.array(atomic_number)
    crystal_name = np.array(crystal_name)
    labels7 = np.array(labels7)
    # atom_labels_list = np.array(atom_labels_list)
    # frac_coords_list = np.array(frac_coords_list)
    
    save_path = os.path.join(dest_path,"%s_%d"%(stage,process_idx))
    # np.save(save_path,{'features':features,'frac_coords':frac_coords_list,'atom_labels':atom_labels_list,'labels230':labels230,'atomic_number':atomic_number,'crystal_name':crystal_name}) # type: ignore
    np.save(save_path,{'features':features,"labels7":labels7,'labels230':labels230,'atomic_number':atomic_number,'crystal_name':crystal_name}) # type: ignore
    logging.info('保存路径:%s'%save_path)
    logging.info('保存的样本数量:%d'%len(labels230))
    return None


def star(args):
    process_wrap(**args)

all_args = []
all_args += [dict(cry_paths=train_data[l:l+args.package_size],process_idx = l/args.package_size,stage='train') for l in range(0,len(train_data),args.package_size)]
all_args += [dict(cry_paths=test_data[l:l+args.package_size],process_idx = l/args.package_size,stage='test') for l in range(0,len(test_data),args.package_size)]


with multiprocessing.Pool(args.worker_num) as pool :
    list(tqdm(pool.imap_unordered(star,all_args),desc='把数据分割打包'))


