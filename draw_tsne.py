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

models = torch.load(args.model_path,map_location=device)
logger.info('-'*15+'args'+'-'*15+'\n%s'%str(args))
logger.info('-'*15+'device'+'-'*15+'\n%s'%str(device))
logger.info('-'*15+'lossfn'+'-'*15+'\n%s'%str(lossfn))
encoder = models['rep']
encoder.eval()
xrd_dataset = XrdData(args.data_path)
dataloader = DataLoader(xrd_dataset,args.batch_size,num_workers=args.num_workers)

# select_class_idxs_05_09 = [1,165,190,195,213]
# select_class_idxs = [138,157,203,209,211] #03->07
# select_class_idxs = [138,157,203,209,211,1,165,190,195,213]
# select_class_idxs = [32,122,124,165,195,213] #raw_good 
# select_class_idxs = [0,15,80,138,157,203,209,211,214,228]
select_class_idxs = [75-1,79-1,81-1,82-1]
xi,xa,xl = [],[],[]
for data in dataloader:
    intensity , angle,labels230 = data[0].type(torch.float).to(device),data[1].type(torch.float).to(device),data[2].to(device)
    # print(intensity.shape,angle.shape)
    for i in range(labels230.shape[0]):
        if labels230[i] in select_class_idxs:
            xi.append(intensity[i].view(1,-1))
            xa.append(angle[i].view(1,-1))
            xl.append(labels230[i])

xi = torch.concat(xi,dim=0)
xa = torch.concat(xa,dim=0)
xl = torch.tensor(xl)

# print(xi.shape,xa.shape,xl.shape)

with torch.no_grad():
    features = encoder(xi,xa).cpu().numpy()

tsne_model = TSNE(n_components=2, perplexity=10, n_iter=2000, random_state=42)
transformed_data = tsne_model.fit_transform(features)

print(list(transformed_data[:, 0]))
print(list(transformed_data[:, 1]))
print(list(xl.cpu().numpy()))

# fig, ax = plt.subplots(1,1, figsize=(20, 8))
# def Normalization2(x):
#     return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]

# in_x = Normalization2(transformed_data[:, 0])
# in_y = Normalization2(transformed_data[:, 1])

# scatter2 = ax.scatter(in_x, in_y, c=xl.cpu().numpy(), cmap='viridis', s=50)
# ax.set_title('t-SNE visualization of Digits dataset')
# fig.colorbar(scatter2, ax=ax)
 
# # 显示图像
# plt.savefig('./temp.png')

# def test():
#     for k in tasks:
#         models[k].eval()
        
#     accs = {
#         'sp':MulticlassAccuracy(230,average='micro').to(device)  ,
#         'cs':MulticlassAccuracy(7,average='micro').to(device)   ,
#         'lt':MulticlassAccuracy(6,average='micro').to(device)   ,
#         'pg':MulticlassAccuracy(32,average='micro').to(device)  ,
#     }
#     lossfns = {k:torch.nn.CrossEntropyLoss() for k in tasks}
    
#     top_k_acc = MulticlassAccuracy(num_classes=230,top_k=args.top_k,average='micro').to(device)
#     f1_score = MulticlassF1Score(num_classes=230,top_k=1).to(device)
#     recall_score = MulticlassRecall(230).to(device)
#     acc_per_class = MulticlassAccuracy(num_classes=230,average=None).to(device)
    
#     total_num = 0 
#     errs = {k:0.0 for k in tasks}  
#     batch_cnt = 0 
#     with torch.no_grad():
#         for file in test_files:
#             xrd_dataset = XrdData(file)
#             dataloader = DataLoader(xrd_dataset,args.batch_size,num_workers=args.num_workers)
#             for data in dataloader:
#                 intensity , angle,labels230,labels7,labels6,labels32 = data[0].type(torch.float).to(device),data[1].type(torch.float).to(device),data[2].to(device),data[4].to(device),data[5].to(device),data[6].to(device)
#                 labels = {
#                     "sp":labels230,
#                     "lt":labels6,
#                     "pg":labels32,
#                     "cs":labels7,
#                 }
#                 rep = models['rep'](intensity,angle)
#                 batch_cnt += 1 
#                 total_num += angle.shape[0]
#                 for t in tasks:
#                     out_t = models[t](rep)
#                     loss_t = lossfns[t](out_t,labels[t])
#                     logits_t = out_t.softmax(dim=1)
#                     errs[t] += loss_t
#                     accs[t](logits_t,labels[t])
#                     if t == 'sp':
#                         top_k_acc(logits_t,labels[t])
#                         f1_score(logits_t,labels[t])
#                         recall_score(logits_t,labels[t])
#                         acc_per_class(logits_t,labels[t])
#     for t in tasks:
#         logger.info("task:%s,err:%s,acc:%s"%(t,str(errs[t]/batch_cnt),str(accs[t].compute().cpu().item())))
#     per_class_acc = list(acc_per_class.compute().cpu().numpy())
#     h_acc,m_acc,t_acc = per_class_acc2hmt_acc(per_class_acc,args.balanced) 
    
#     logger.info("sp_f1:%s\n,sp_recall:%s,sp_top%d_acc:%s\nsp_head_acc:%s;sp_mid_acc:%s;sp_tail_acc:%s"%(
#         str(f1_score.compute().cpu().item()),
#         str(recall_score.compute().cpu().item()),
#         args.top_k,
#         str(top_k_acc.compute().cpu().item()),
#         str(h_acc),
#         str(m_acc),
#         str(t_acc),
#     ))
        
#     logger.info("per_class_acc:%s"%per_class_acc)

# if __name__ == '__main__':
#     test()
    
    
    