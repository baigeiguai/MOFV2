from pymatgen.core import Structure # ,Lattice, Molecule
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import argparse 
import os 
import numpy as np
import logging
from torch import  multiprocessing as mp
from tqdm import tqdm


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
    
    


ANGLE_START = 5
ANGLE_END = 90
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,required=True)
parser.add_argument('--dest_path',type=str,required=True)
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

COORDINATE_NUMBER_LIMIT=500
def process_cif(file_path):
    try:
        structure  = Structure.from_file(file_path,primitive=True)
        space_group = structure.get_space_group_info()[1]-1
        assert (space_group+1) in space_group_map_dict
        crystal_system = space_group_map_dict[space_group+1]-1
        cart_coords = structure.cart_coords[:COORDINATE_NUMBER_LIMIT]
        atomic_numbers = structure.atomic_numbers[:COORDINATE_NUMBER_LIMIT]
        lattice =  structure.lattice.metric_tensor
    except:
        logging.error('file_path:%s\n'%file_path)
        return 
    dest_path = os.path.join(args.dest_path,str(space_group)) 
    crystal_name = os.path.basename(file_path).split('.')[0]
    dest_path = os.path.join(dest_path,crystal_name + '.npy')
    
    print(dest_path)
    if  os.path.exists(dest_path):
        return 
    try :
        xrd = XRDCalculator(wavelength=1.54056,symprec=0.1) # type: ignore 
        pattern = xrd.get_pattern(structure,scaled=True,two_theta_range=(ANGLE_START,ANGLE_END))
        # atomic_number = len(structure.atomic_numbers)
    except:
        logging.error('file_path:%s\n'%file_path)
        return 
    np.save(dest_path,{'pattern':pattern,
                       "crystal_system":crystal_system,
                       'space_group':space_group,
                       'atomic_numbers':atomic_numbers,
                       'crystal_name':crystal_name,
                       "cart_coords":cart_coords,
                       "lattice":lattice,
                       }) # type: ignore 
    return None

def star(args):
    t = process_cif(**args)
    return t 

logging.basicConfig(filename='poscar2npy.err')

if __name__ =='__main__':
    with mp.Pool(args.worker_num) as pool:
        proc_args = [dict(file_path=f) for f in file_paths]
        # print(proc_args)
        list(tqdm(pool.imap_unordered(star,proc_args),desc=f'poscar2npy ing. '
                        f'This can take a moment. Using {args.worker_num} workers', total=len(proc_args)))

