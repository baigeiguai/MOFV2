import numpy as np 
import os 
import argparse
from ast import parse
from tqdm import tqdm 
from torch import multiprocessing

parser = argparse.ArgumentParser('remove coord data')
parser.add_argument('--data_path',type=str,required=True)
parser.add_argument('--to_path',type=str,required=True)
parser.add_argument('--worker_num',type=int,default=25)

args = parser.parse_args()

all_args = []

for i in range(230):
    src_path = os.path.join(args.data_path,str(i))
    dst_path = os.path.join(args.to_path,str(i))
    for file in os.listdir(src_path):
        file_path = os.path.join(src_path,file)
        cry_name = os.path.basename(file_path)
        all_args.append(dict(src_path=file_path,to_path=os.path.join(dst_path,cry_name)))
    #     break
    # break



def process(src_path,to_path):
    data = np.load(src_path,allow_pickle=True)
    pattern = data.item().get('pattern')
    crystal_system =  data.item().get('crystal_system')
    space_group = data.item().get('space_group')
    crystal_name = data.item().get('crystal_name')
    atomic_number = data.item().get('atomic_number')
    np.save(to_path,{'pattern':pattern,'crystal_system':crystal_system,'space_group':space_group,'crystal_name':crystal_name,'atomic_number':atomic_number})



def star(args):
    process(**args)
    
with multiprocessing.Pool(args.worker_num) as pool :
    list(tqdm(pool.imap_unordered(star,all_args),desc=''))