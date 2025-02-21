

# [0, 1, 3, 4, 6, 8, 10, 11, 12, 13, 14, 17, 18, 19, 28, 30, 32, 35, 42, 44, 51, 53, 55, 56, 57, 59, 60, 61, 62, 69, 75, 77, 81, 84, 85, 86, 87, 91, 95, 113, 121, 141, 143, 144, 145, 146, 147, 151, 153, 160, 164, 165, 166, 168, 169, 172, 175, 197, 204, 224]
from torch.utils.data import DataLoader
from utils.dataset import XrdData
import os 
import torch 
import argparse 
from utils.logger import Log
from torchmetrics.classification import MulticlassAccuracy,MulticlassF1Score,MulticlassRecall
from utils.convert import per_class_acc2hmt_acc
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np 

parser = argparse.ArgumentParser()

parser.add_argument('--data_path',type=str,required=True)
parser.add_argument('--batch_size',type=int,default=128)
parser.add_argument('--model_path',type=str,required=True)
parser.add_argument('--device',type=str,default="0")
parser.add_argument('--mode',type=str,choices=['train','test'],default="test")
parser.add_argument('--top_k',type=int,default=5)
parser.add_argument('--parallel_model',type=bool,default=False)
parser.add_argument('--test_name',type=str,required=True)
parser.add_argument("--num_workers",type=int,default=20)
parser.add_argument("--balanced",type=int,required=True)

args = parser.parse_args()


tasks = ['sp','lt','cs','pg']

log = Log(__name__,'log/test','test_%s'%(args.test_name))
args.log_name = log.log_name
logger = log.get_log()

device = torch.device("cuda:%s"%args.device if torch.cuda.is_available() else 'cpu')
lossfn = torch.nn.CrossEntropyLoss()

file_paths = os.listdir(args.data_path)
train_files,test_files = [os.path.join(args.data_path,f)  for f in file_paths if f.startswith('train')],[os.path.join(args.data_path,f) for f in file_paths if f.startswith('test')]

models = torch.load(args.model_path,map_location=device)
logger.info('-'*15+'args'+'-'*15+'\n%s'%str(args))
logger.info('-'*15+'device'+'-'*15+'\n%s'%str(device))
logger.info('-'*15+'lossfn'+'-'*15+'\n%s'%str(lossfn))
encoder = models['rep'].to(device)
encoder.eval()

# select_class_idxs = [0, 1, 3, 4, 6, 8, 10, 11, 12, 13, 14, 17, 18, 19, 28, 30, 32, 35, 42, 44, 51, 53, 55, 56, 57, 59, 60, 61, 62, 69, 75, 77, 81, 84, 85, 86, 87, 91, 95, 113, 121, 141, 143, 144, 145, 146, 147, 151, 153, 160, 164, 165, 166, 168, 169, 172, 175, 197, 204, 224]
select_class_idxs = [75-1,79-1,81-1,82-1]

cls2xi = {i:[] for i in select_class_idxs}
cls2xa = {i:[] for i in select_class_idxs}
cls2sp = {i:[] for i in select_class_idxs}
limit = 50


for file in train_files:
    xrd_dataset = XrdData(file)
    dataloader = DataLoader(xrd_dataset,args.batch_size,num_workers=args.num_workers)
    for data in dataloader:
        intensity , angle,labels230 = data[0].type(torch.float),data[1].type(torch.float),data[2]
        for i in range(labels230.shape[0]):
            sp = labels230[i].item()
            if sp in select_class_idxs and len(cls2sp[sp])<limit:
                cls2sp[sp].append(sp)
                cls2xi[sp].append(intensity[i].view(1,-1))
                cls2xa[sp].append(angle[i].view(1,-1))
                
for k in cls2xi:
    cls2xi[k] = torch.concat(cls2xi[k],dim=0)

for k in cls2xa:
    cls2xa[k] = torch.concat(cls2xa[k],dim=0)
    
labels = []
for k in cls2sp:
    labels += cls2sp[k]
lll = {v:i for i,v in enumerate(select_class_idxs)}
labels= [lll[i] for i in labels]

features = None
with torch.no_grad():
    for k in cls2xi:
        intensity,angle = cls2xi[k].to(device),cls2xa[k].to(device)
        if features is None:
            features = encoder(intensity,angle)
        else:
            features = torch.concat([features,encoder(intensity,angle)],dim=0)

features = features.cpu().numpy()

tsne_model = TSNE(n_components=2, perplexity=10, n_iter=2000, random_state=42)
transformed_data = tsne_model.fit_transform(features)

def Normalization2(x):
    return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]

in_x = Normalization2(transformed_data[:, 0])
in_y = Normalization2(transformed_data[:, 1])

np.save('draw_train_tsne_case_IP4_4.npy',{
    'x':in_x,
    'y':in_y,
    'c':labels})

# fig, ax = plt.subplots(1,1, figsize=(12, 8))
# scatter2 = ax.scatter(in_x, in_y, c=labels, cmap='viridis', s=20)
# ax.set_title('t-SNE visualization of Digits dataset')
# fig.colorbar(scatter2, ax=ax)

# # 显示图像
# plt.savefig('./temp_train.png')


# select_class_idxs_05_09 = [1,165,190,195,213]
# select_class_idxs = [138,157,203,209,211] #03->07
# select_class_idxs = [138,157,203,209,211,1,165,190,195,213]
# select_class_idxs = [32,122,124,165,195,213] #raw_good 
# select_class_idxs = [0,15,80,138,157,203,209,211,214,228]
# xi,xa,xl = [],[],[]
# for data in dataloader:
#     intensity , angle,labels230 = data[0].type(torch.float).to(device),data[1].type(torch.float).to(device),data[2].to(device)
#     # print(intensity.shape,angle.shape)
#     for i in range(labels230.shape[0]):
#         if labels230[i] in select_class_idxs:
#             xi.append(intensity[i].view(1,-1))
#             xa.append(angle[i].view(1,-1))
#             xl.append(labels230[i])

# xi = torch.concat(xi,dim=0)
# xa = torch.concat(xa,dim=0)
# xl = torch.tensor(xl)

# print(xi.shape,xa.shape,xl.shape)

# with torch.no_grad():
#     features = encoder(xi,xa).cpu().numpy()

# tsne_model = TSNE(n_components=2, perplexity=10, n_iter=2000, random_state=42)
# transformed_data = tsne_model.fit_transform(features)

# print(list(transformed_data[:, 0]))
# print(list(transformed_data[:, 1]))
# print(list(xl.cpu().numpy()))

# def Normalization2(x):
#     return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]

# in_x = Normalization2(transformed_data[:, 0])
# in_y = Normalization2(transformed_data[:, 1])

# fig, ax = plt.subplots(1,1, figsize=(20, 8))
# scatter2 = ax.scatter(in_x, in_y, c=xl.cpu().numpy(), cmap='viridis', s=50)
# ax.set_title('t-SNE visualization of Digits dataset')
# fig.colorbar(scatter2, ax=ax)
 
# # 显示图像
# plt.savefig('./temp.png')

    
    
    