import os 
import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,required=True)
parser.add_argument('--to_path',type=str,required=True)
parser.add_argument('--mode',type=str,default='train')

args = parser.parse_args()

sp2data = {sp_idx:[] for sp_idx in range(230)}

for file in os.listdir(args.data_path):
    if not (file.endswith('npy') and file.startswith(args.mode)):
        continue
    file_path = os.path.join(args.data_path,file)
    data = np.load(file_path,allow_pickle=True).item()
    features = data.get('features')
    sp = data.get('labels230')
    L = features.shape[0]
    for i in range(L):
        sp2data[sp[i]].append(features[i])
    print(file,'done!')

for i in range(230):
    if len(sp2data[i]) > 0 :
        np.save(os.path.join(args.to_path,str(i),str(i)),np.array(sp2data[i]),allow_pickle=True)

    