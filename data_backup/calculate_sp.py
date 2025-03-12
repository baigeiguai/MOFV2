from pymatgen.core import Structure 
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import argparse 
import os 
import numpy as np
import logging
from torch import  multiprocessing as mp
from tqdm import tqdm


ANGLE_START = 5
ANGLE_END = 90
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,required=True)
parser.add_argument('--dest_path',type=str,required=True)
parser.add_argument('--worker_num',type=int,required=True)
parser.add_argument('--package_size',type=int,default=1000)

args = parser.parse_args()

file_paths = []
for space in os.listdir(args.data_path):
    new_path = os.path.join(args.data_path,space)
    if not os.path.isdir(new_path):
        continue
    for f in os.listdir(new_path):
        file_path = os.path.join(new_path,f)
        if not file_path.endswith('.POSCAR'):
            continue
        file_paths.append(file_path)


def process_cif(file_paths,process_idx):
    d = {}
    for file_path in file_paths:
        try:
            structure  = Structure.from_file(file_path,primitive=True)
            space_group_idx = structure.get_space_group_info()[1]-1
            space_group_name = structure.get_space_group_info()[0]
            d[space_group_idx] = space_group_name
        except:
            logging.error('file_path:%s\n'%file_path)
            return 
    dest_path = os.path.join(args.dest_path,str(process_idx)) 
    print(args.dest_path,dest_path,process_idx)
    np.save(dest_path,d)

    return None

def star(args):
    t = process_cif(**args)
    return t 

logging.basicConfig(filename='poscar2npy.err')

if __name__ =='__main__':
    with mp.Pool(args.worker_num) as pool:
        # all_args += [dict(cry_paths=train_data[l:l+args.package_size],process_idx = l/args.package_size,stage='train') for l in range(0,len(train_data),args.package_size)]
        
        proc_args = [dict(file_paths=file_paths[l:l+args.package_size],process_idx = l//args.package_size) for l in range(0,len(file_paths),args.package_size)]
        # print(proc_args)
        list(tqdm(pool.imap_unordered(star,proc_args),desc=f'poscar2npy ing. '
                        f'This can take a moment. Using {args.worker_num} workers', total=len(proc_args)))

