import os 
import heapq
import argparse
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,required=True)
parser.add_argument("--maxi_number",type=int,default=100)
args = parser.parse_args()
max_list =  []
mini_list = [] 

for space_group_idx in range(230):
    print("doing ",space_group_idx)
    space_group_path = os.path.join(args.data_path,str(space_group_idx))
    for file in os.listdir(space_group_path):
        file_path = os.path.join(space_group_path,file)
        print(space_group_idx,file_path)
        data = np.load(file_path,allow_pickle=True)
        l = len(data.item().get('frac_coords'))
        heapq.heappush(max_list,l)
        heapq.heappush(mini_list,-l)
        if len(max_list) > args.maxi_number :
            heapq.heappop(max_list)
        if len(mini_list) > args.maxi_number:
            heapq.heappop(mini_list)
        
print("minis:",heapq.nlargest(args.maxi_number,mini_list))
print("maxis:",heapq.nsmallest(args.maxi_number,max_list))
