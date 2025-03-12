# from pymatgen.io.cif import CifParser
from pymatgen.core import Structure # ,Lattice, Molecule
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
parser.add_argument('--worker_num',type=int,required=True)

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

def process_cif(file_path):
    try:
        structure  = Structure.from_file(file_path,primitive=True)
        return len(structure.cart_coords)
    except:
        logging.error('file_path:%s\n'%file_path)
        return 0
    
    
    return 0

def star(args):
    t = process_cif(**args)
    return t 

logging.basicConfig(filename='poscar2npy.err')

if __name__ == '__main__':
    with mp.Pool(args.worker_num) as pool:
        proc_args = [dict(file_path=f) for f in file_paths]
        # print(proc_args)
        out = list(tqdm(pool.imap_unordered(star,proc_args),desc=f'poscar2npy ing. '
                        f'This can take a moment. Using {args.worker_num} workers', total=len(proc_args)))
        out = sorted(out,reverse=True)
        print(out)
        print("out_10:",out[10])
        print("out_100:",out[100])
        print("out_1000:",out[1000])
    

