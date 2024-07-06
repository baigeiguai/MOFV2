from pymatgen.io.cif import CifParser
from pymatgen.core import Lattice, Structure, Molecule
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

def process_cif(file_path):
    try:
        structure  = Structure.from_file(file_path,primitive=True)
        structure = structure.make_supercell([2,2,2])
        space_group = structure.get_space_group_info()[1]-1
        # print("space group",space_group)
        assert (space_group+1) in space_group_map_dict
        crystal_system = space_group_map_dict[space_group+1]-1
        # frac_coords = structure.frac_coords
        # atom_labels = structure.atomic_numbers
    except:
        logging.error('file_path:%s\n'%file_path)
        return 
    try :
        xrd = XRDCalculator(wavelength=1.54056,symprec=0.1) # type: ignore 
        pattern = xrd.get_pattern(structure,scaled=True,two_theta_range=(ANGLE_START,ANGLE_END))
        atomic_number = len(structure.atomic_numbers)
    except:
        logging.error('file_path:%s\n'%file_path)
        return 

    return [space_group,crystal_system,atomic_number],pattern

def process_cif2(file_path):
    try:
        structure  = Structure.from_file(file_path,primitive=True)
        space_group = structure.get_space_group_info()[1]-1
        # print("space group",space_group)
        assert (space_group+1) in space_group_map_dict
        crystal_system = space_group_map_dict[space_group+1]-1
        # frac_coords = structure.frac_coords
        # atom_labels = structure.atomic_numbers
    except:
        logging.error('file_path:%s\n'%file_path)
        return 
    try :
        xrd = XRDCalculator(wavelength=1.54056,symprec=0.1) # type: ignore 
        pattern = xrd.get_pattern(structure,scaled=True,two_theta_range=(ANGLE_START,ANGLE_END))
        atomic_number = len(structure.atomic_numbers)
    except:
        logging.error('file_path:%s\n'%file_path)
        return 

    return [space_group,crystal_system,atomic_number],pattern

def judge(a,b):
    L = len(a)
    maxi = 0
    res = True
    for i in range(L):
        if a[i] != b[i] :
            # print('diff',i,a[i],b[i])
            res = False
            maxi = max(maxi,abs(a[i]-b[i]))
    print("maxi",maxi)
    return res 

a1,b1 = process_cif('/home/ylh/code/MyExps/MOF/test_data/test_data/test/AETXCO01.Ama2.POSCAR')
a2,b2 = process_cif2('/home/ylh/code/MyExps/MOF/test_data/test_data/test/AETXCO01.Ama2.POSCAR')
print(a1,a2,len(b1.x),len(b2.x))
print(b1.x[:10],b2.x[:10])
print(b1.y[:10],b2.y[:10])
print(judge(a1,a2),judge(b1.x,b2.x),judge(b1.y,b2.y))