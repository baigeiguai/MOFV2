import torch 
import  numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',type=str)


args = parser.parse_args()
models = torch.load(args.model_path,map_location=torch.device("cpu"))

tasks = ['sp','lt','cs','pg']
data = {t: models[t].cls[0].weight.detach().cpu().numpy() for t in tasks}
np.save('cls_weight.npy',data)