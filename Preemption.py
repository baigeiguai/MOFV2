import torch 
import time
import random 

device = torch.device("cuda:7")

a = torch.rand(2048,8192,1024).to(device)

b = 1 
while (2>1):
    b *= -1 
    a *= b
    time.sleep(random.random())