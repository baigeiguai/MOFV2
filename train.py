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

if args.model_path is None :
   # from models.XAt_base import XrdAttentionBase 
   #  model = XrdAttentionBase().to(device)
    
    # from models.SConv import SResTcn
    # model = SResTcn().to(device)
    
    from models.RawEmbedConv import RawEmbedConv
    model = RawEmbedConv().to(device)
    
else :
    model = torch.load(args.model_path,map_location=device)

if len(device_list) >  1 :
    model = torch.nn.DataParallel(model,device_list).to(device)

lossfn = torch.nn.CrossEntropyLoss().to(device)

if not os.path.exists(args.model_save_path):
    os.mkdir(args.model_save_path)

with open(os.path.join(args.model_save_path,'config.json'),'w') as json_file :
    json_file.write(json.dumps(vars(args)))
    
model_save_path = os.path.join(args.model_save_path,args.train_name)

optimizer = torch.optim.AdamW(params=model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
scheduler_T = args.epoch_num-args.start_scheduler_step if args.scheduler_T is None else args.scheduler_T
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,scheduler_T,args.min_learning_rate)

logger.info('-'*15+'args'+'-'*15+'\n'+str(args))
logger.info('-'*15+'model'+'-'*15+'\n'+str(model))
logger.info('-'*15+'device'+'-'*15+'\n'+str(device))
logger.info('-'*15+'optimizer'+'-'*15+'\n'+str(optimizer))
logger.info('-'*15+'lossfn'+'-'*15+'\n'+str(lossfn))
logger.info('-'*15+'seed'+'-'*15+'\n'+str(now_seed))

file_paths = os.listdir(args.data_path)
train_files,test_files = [os.path.join(args.data_path,f)  for f in file_paths if f.startswith('train')],[os.path.join(args.data_path,f) for f in file_paths if f.startswith('test')]

writer = SummaryWriter(log_dir='./hh_board_dir/%s'%args.log_name)


def train():
    max_acc = 0 
    mini_err = 1e9 
    for epoch_idx in range(args.epoch_num):
        logger.info('-'*15+'epoch '+str(epoch_idx+1)+'-'*15+'\nlr: '+str(lr_scheduler.get_lr()))
        total_num = 0.0
        total_err = 0.0 
        batch_cnt = 0
        # model.train()
        for file in train_files:
            xrd_dataset = XrdData(file)
            dataloader = DataLoader(xrd_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
            for data in dataloader:
                optimizer.zero_grad()
                intensity,angle,labels230 = data[0].type(torch.float).to(device),data[1].to(device),data[2].to(device)
                out = model(intensity,angle)
                logits = out 
                # logits,hkl = out[0],out[1]
                error = lossfn(logits,labels230)
                error.backward()
                optimizer.step()
                total_num += labels230.shape[0]
                batch_cnt += 1 
                total_err += error.item()
        logger.info('[training]total_num: '+str(total_num)+',error: '+str(total_err/batch_cnt))
        test_acc,test_err = test()
        writer.add_scalar("train/acc",test_acc,epoch_idx+1)
        writer.add_scalar("train/err",test_err,epoch_idx+1)
        if epoch_idx >= args.start_scheduler_step:
            lr_scheduler.step()
        
        if mini_err > test_err :
            mini_err = test_err 
            max_acc = max(max_acc,test_acc)
            torch.save(model if not len(device_list)>1  else model.module,model_save_path+'_epoch_%d'%(epoch_idx+1)+'.pth')
        elif max_acc < test_acc :
            max_acc = test_acc 
            torch.save(model if not len(device_list)>1  else model.module,model_save_path+'_epoch_%d'%(epoch_idx+1)+'.pth')
    
        
def test():
    model.eval()
    total_acc =  MulticlassAccuracy(args.class_num,average='micro').to(device)    
    total_num = 0 
    total_err = 0.0 
    batch_cnt = 0 
    with torch.no_grad():
        for file in train_files:
            xrd_dataset = XrdData(file)
            dataloader = DataLoader(xrd_dataset,args.batch_size,num_workers=args.num_workers)
            for data in dataloader:
                intensity , angle,labels230 = data[0].type(torch.float).to(device),data[1].type(torch.float).to(device),data[2].to(device)
                # print('labels230 shape',labels230.shape)
                out = model(intensity,angle)
                raw_logits = out 
                # raw_logits,hkl = out[0],out[1]
                err = lossfn(raw_logits,labels230)
                total_err += err.item()
                logits = raw_logits.softmax(dim=1)
                total_num += labels230.shape[0]
                total_acc(logits,labels230)
                batch_cnt += 1 
    
    total_acc_val = total_acc.compute().cpu().item()
    logger.info('[testing]total_number: '+str(total_num)+',error: '+str(total_err/batch_cnt)+',total_acc: '+str(total_acc_val))
    return total_acc_val,total_err/batch_cnt
                
if __name__ == '__main__':
    train()
    writer.close()
