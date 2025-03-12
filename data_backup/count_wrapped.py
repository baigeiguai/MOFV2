import numpy as np 
import os 
import argparse 
parser = argparse.ArgumentParser()

parser.add_argument('--data_path',type=str,required=True)
parser.add_argument('--mood',type=str,required=True)

args = parser.parse_args()
cnt  = [0 for i in range(230)]

for file in os.listdir(args.data_path):
    if not file.startswith(args.mood):
        continue
    file_path = os.path.join(args.data_path,file)
    data = np.load(file_path,allow_pickle=True)
    for j in data.item().get('labels230'):
        cnt[j] += 1
    
print(cnt)
    