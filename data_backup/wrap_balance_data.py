import os 
import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,required=True)
parser.add_argument('--to_path',type=str,required=True)
parser.add_argument('--mode',type=str,default='test')
parser.add_argument('--per_class_num',type=int,default=10)

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
        if len(sp2data[sp[i]])<args.per_class_num:
            sp2data[sp[i]].append(features[i])
    print(file,'done!')

all_labels230 = []
all_features  = []
for i in range(230):
    if len(sp2data[i]) >= args.per_class_num :
        all_labels230+=[i for k in range(len(sp2data[i]))]
        all_features += sp2data[i]

all_features = np.array(all_features)
all_labels230 = np.array(all_labels230)        

if not os.path.exists(args.to_path):
    os.mkdir(args.to_path)
    
np.save(os.path.join(args.to_path,'test_0.npy'),{
    'features':all_features,
    'labels230':all_labels230 })


    