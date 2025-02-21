import logging
from torch.utils.data import DataLoader
import json
import os
import torch 
import argparse
from utils.init import seed_torch
from utils.logger import Log
from torch.utils.tensorboard.writer import SummaryWriter
from utils.dataset import XrdData
from torchmetrics.classification import MulticlassAccuracy
from utils.ema import EMA

from models.XRDecouplerEncoder import XRDecouplerEncoder 
from models.XRDecouplerCls import *
from utils.normalization import gradient_normalizers,MinNormSolver


torch.backends.cudnn.enabled = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,required=True)
parser.add_argument('--train_name',type=str,required=True)
parser.add_argument('--model_path',type=str)
parser.add_argument('--learning_rate',type=float,default=0.01)
parser.add_argument('--min_learning_rate',type=float,default=0.001)
parser.add_argument('--start_scheduler_step',type=int,default=0)
parser.add_argument('--weight_decay',type=float,default=1e-5)
parser.add_argument('--momentum',type=float,default=0.99)
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument("--class_num",type=int,default=230)
parser.add_argument("--epoch_num",type=int,default=200)
parser.add_argument("--model_save_path",type=str,required=True)
parser.add_argument("--device",type=str,default="0")
parser.add_argument("--scheduler_T",type=int)
parser.add_argument("--num_workers",type=int,default=20)
args = parser.parse_args()
log = Log(__name__,file_dir='log/train/',log_file_name='train_%s'%(args.train_name))
args.log_name = log.log_name
logger = log.get_log()

now_seed = 3407
seed_torch(now_seed)

device_list = [int(i) for i in args.device.split(',')]
device = torch.device('cuda:%d'%device_list[0] if  torch.cuda.is_available() else 'cpu')

tasks = ['sp','lt','cs','pg']
lossfns = {k:torch.nn.CrossEntropyLoss() for k in tasks}
if args.model_path:
    models = torch.load(args.model_path,map_location=device)
    for k in models:
        models[k] = models[k].to(device)
else:
    models = {
        'rep': XRDecouplerEncoder().to(device),
        'sp': ClsSp().to(device),
        'lt': ClsLt().to(device),
        'cs': ClsCs().to(device),
        'pg': ClsPg().to(device),
    }

if len(device_list) > 1 :
    for k in models:
        models[k] = torch.nn.DataParallel(models[k],device_list).to(device)
    

model_params = []
for m in models:
    model_params += models[m].parameters()
    
optimizer = torch.optim.Adam(model_params,lr=args.learning_rate)
scheduler_T = args.epoch_num-args.start_scheduler_step if args.scheduler_T is None else args.scheduler_T
lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,1,0.002,args.epoch_num)
if not os.path.exists(args.model_save_path):
    os.mkdir(args.model_save_path)

with open(os.path.join(args.model_save_path,'config.json'),'w') as json_file :
    json_file.write(json.dumps(vars(args)))
    
model_save_path = os.path.join(args.model_save_path,args.train_name)

logger.info('-'*15+'args'+'-'*15+'\n'+str(args))
logger.info('-'*15+'device'+'-'*15+'\n'+str(device))
logger.info('-'*15+'optimizer'+'-'*15+'\n'+str(optimizer))
logger.info('-'*15+'seed'+'-'*15+'\n'+str(now_seed))

file_paths = os.listdir(args.data_path)
train_files,test_files = [os.path.join(args.data_path,f)  for f in file_paths if f.startswith('train')],[os.path.join(args.data_path,f) for f in file_paths if f.startswith('test')]

writer = SummaryWriter(log_dir='./board_dir/%s'%args.log_name)


def train():
    max_acc = 0 
    mini_err = 1e9 
    for epoch_idx in range(args.epoch_num):
        logger.info('-'*15+'epoch '+str(epoch_idx+1)+'-'*15+'\nlr: '+str(lr_scheduler.get_lr()))
        total_num = 0.0
        total_err = {t:0.0 for t in tasks}
        batch_cnt = 0
        for m in models :
            models[m].train()
        for file in train_files:
            xrd_dataset = XrdData(file)
            dataloader = DataLoader(xrd_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
            for data in dataloader:
                optimizer.zero_grad()
                intensity,angle,labels230,labels7,labels6,labels32 = data[0].type(torch.float).to(device),data[1].to(device),data[2].to(device),data[4].to(device),data[5].to(device),data[6].to(device)
                labels = {
                    "sp":labels230,
                    "lt":labels6,
                    "pg":labels32,
                    "cs":labels7,
                }
                loss_data = {}
                grads = {}
                scale = {}
                rep = models['rep'](intensity,angle)
                rep_variable = torch.autograd.Variable(rep.data.clone(),requires_grad=True)
                for t in tasks:
                    optimizer.zero_grad()
                    out_t  = models[t](rep_variable)    
                    loss = lossfns[t](out_t,labels[t])
                    loss_data[t] = loss.data
                    loss.backward()
                    grads[t] = []
                    grads[t].append(torch.autograd.Variable(rep_variable.grad.data.clone(),requires_grad=False))
                    rep_variable.grad.data.zero_()
                
                gn = gradient_normalizers(grads,loss_data)
                for t in tasks:
                    for gr_i in range(len(grads[t])):
                        grads[t][gr_i] = grads[t][gr_i] / gn[t]
                
                sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
                for i,t in enumerate(tasks):
                    scale[t] = float(sol[i])
                    
                optimizer.zero_grad()
                rep = models['rep'](intensity,angle)
                for i,t in enumerate(tasks):
                    out_t = models[t](rep)
                    loss_t = lossfns[t](out_t,labels[t])
                    loss_data[t] = loss_t.data
                    if i>0:
                        loss += scale[t]*loss_t
                    else:
                        loss = scale[t]*loss_t
                    
                loss.backward()
                optimizer.step()
                for t in tasks:
                    total_err[t] += loss_data[t].item()
                batch_cnt += 1 
                total_num += angle.shape[0]
        logger.info("-"*15+"[trainning]"+"-"*15)
        for t in tasks:
            logger.info("task:%s,err:%s"%(t,total_err[t]/batch_cnt))
        test_acc,test_err = test(epoch_idx)
        
        if epoch_idx >= args.start_scheduler_step:
            lr_scheduler.step()
        if mini_err > test_err :
            mini_err = test_err 
            max_acc = max(max_acc,test_acc)
            torch.save(models ,model_save_path+'_epoch_%d'%(epoch_idx+1)+'.pth') 
        elif max_acc < test_acc :
            max_acc = test_acc 
            torch.save(models ,model_save_path+'_epoch_%d'%(epoch_idx+1)+'.pth')
        
        if epoch_idx %25 == 0 :
            os.system("nvidia-smi")
    
        
def test(epoch_idx):
    for m in models:
        models[m].eval()
    accs = {
        'sp':MulticlassAccuracy(args.class_num,average='micro').to(device)  ,
        'cs':MulticlassAccuracy(7,average='micro').to(device)   ,
        'lt':MulticlassAccuracy(6,average='micro').to(device)   ,
        'pg':MulticlassAccuracy(32,average='micro').to(device)  ,
    }
    
    total_num = 0 
    errs = {k:0.0 for k in tasks}  
    batch_cnt = 0 
    with torch.no_grad():
        for file in train_files:
            xrd_dataset = XrdData(file)
            dataloader = DataLoader(xrd_dataset,args.batch_size,num_workers=args.num_workers)
            for data in dataloader:
                intensity , angle,labels230,labels7,labels6,labels32 = data[0].type(torch.float).to(device),data[1].type(torch.float).to(device),data[2].to(device),data[4].to(device),data[5].to(device),data[6].to(device)
                labels = {
                    "sp":labels230,
                    "lt":labels6,
                    "pg":labels32,
                    "cs":labels7,
                }
                rep = models['rep'](intensity,angle)
                batch_cnt += 1 
                total_num += angle.shape[0]
                for t in tasks:
                    out_t = models[t](rep)
                    loss_t = lossfns[t](out_t,labels[t])
                    logits_t = out_t.softmax(dim=1)
                    errs[t] += loss_t
                    accs[t](logits_t,labels[t])
                    
    logger.info("-"*15+"[testing]"+"-"*15)
    for t in tasks:
        writer.add_scalar("train/%s_err"%t,errs[t]/batch_cnt,epoch_idx)
        writer.add_scalar("train/%s_acc"%t,accs[t].compute().cpu().item(),epoch_idx)
        logger.info("task:%s,err:%s,acc:%s"%(t,str(errs[t]/batch_cnt),str(accs[t].compute().cpu().item())))
            
    
    return accs['sp'].compute().cpu().item(),errs['sp']/batch_cnt
                
if __name__ == '__main__':
    train()
    writer.close()