import logging
import torch.utils
from torch.utils.data import DataLoader
import json
import os
import torch 
import argparse
from utils.init import seed_torch
from utils.logger import Log
from torch.utils.tensorboard.writer import SummaryWriter
from utils.dataset import AllXrdDataset
from torchmetrics.classification import MulticlassAccuracy
from utils.ema import EMA
from losses.SupConLoss import SupConLoss
from utils.cluster import kmeans

torch.backends.cudnn.enabled = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,required=True)
parser.add_argument('--train_name',type=str,required=True)
parser.add_argument('--model_path',type=str)
parser.add_argument('--learning_rate',type=float,default=0.01)
parser.add_argument('--min_learning_rate',type=float,default=5e-4)
parser.add_argument('--start_scheduler_step',type=int,default=0)
parser.add_argument('--weight_decay',type=float,default=4e-5)
parser.add_argument('--momentum',type=float,default=0.99)
parser.add_argument('--batch_size',type=int,default=512)
parser.add_argument("--class_num",type=int,default=230)
parser.add_argument("--epoch_num",type=int,default=150)
parser.add_argument("--model_save_path",type=str,required=True)
parser.add_argument("--device",type=str,default="0")
parser.add_argument("--scheduler_T",type=int)
parser.add_argument("--num_workers",type=int,default=20)
parser.add_argument("--cluster_limit",type=int,default=200)
parser.add_argument("--warm_epoch",type=int,default=10)
parser.add_argument("--epoch_step",type=int,default=10)
args = parser.parse_args()

log = Log(__name__,file_dir='log/train/',log_file_name='train_%s'%(args.train_name))
args.log_name = log.log_name
logger = log.get_log()

now_seed = 3407
seed_torch(now_seed)

device_list = [int(i) for i in args.device.split(',')]
device = torch.device('cuda:%d'%device_list[0] if  torch.cuda.is_available() else 'cpu')


train_dataset = AllXrdDataset(args.data_path)
cls_num_list = train_dataset.cls_cnt
# print("cls num list:%s"%str(cls_num_list))
logger.info("cls num list:%s"%str(cls_num_list))
args.cls_num_list = cls_num_list

train_loader_cluster = DataLoader(
    train_dataset,batch_size=args.batch_size,shuffle=False,
    num_workers=args.num_workers,pin_memory=True
)

cluster_number = [t//max(min(cls_num_list),args.cluster_limit) for t in cls_num_list]
for index,value in enumerate(cluster_number):
    if value == 0 and cls_num_list[index] > 0  :
        cluster_number[index] = 1 

# print("cluster_number:",cluster_number)
logger.info("cluster_number:%s"%str(cluster_number))

train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True)


if args.model_path is None :
    from models.HopeV1_Sub import HopeV1_Sub 
    model = HopeV1_Sub(sum(cluster_number)).to(device)
else :
    model = torch.load(args.model_path,map_location=device)

if len(device_list) >  1 :
    model = torch.nn.DataParallel(model,device_list).to(device)

ema = EMA(model=model, decay=0.995)
ema.register()
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


writer = SummaryWriter(log_dir='./board_dir/%s'%args.log_name)


def train():

    mini_err = 1e9
    max_acc = 0.0    
    for epoch_idx in range(args.epoch_num):
        logger.info('-'*15+'epoch '+str(epoch_idx+1)+'-'*15+'\nlr: '+str(lr_scheduler.get_lr()))
        avg_error = 0.0
        batch_cnt = 0   
        total_num = 0.0
        if epoch_idx >= args.warm_epoch:
            if epoch_idx % args.epoch_step == 0 :
                logger.info("cluster starts")
                targets = cluster(train_loader_cluster,cluster_number)
                train_dataset.new_labels230 = targets
                logger.info("cluster ends")
        model.train()            
        for idx,(intensity,angle,labels230,cluster_label) in enumerate(train_loader):
            optimizer.zero_grad()
            intensity,angle,labels230,cluster_label = intensity.to(device).type(torch.float32),angle.to(device).type(torch.float32),labels230.to(device).type(torch.long),cluster_label.to(device).type(torch.long)
            _,sp_cls,cluster_cls = model(intensity,angle)
            error = lossfn(sp_cls,labels230)
            if cluster_label[0] != -1 :
                error += lossfn(cluster_cls,cluster_label)
            
            error.backward()
            avg_error += error.item()            
            optimizer.step()
            ema.update()
            batch_cnt+=1 
            total_num += labels230.shape[0]
        test_acc,test_err = test()
        
        if mini_err > test_err :
            mini_err = test_err 
            max_acc = max(max_acc,test_acc)
            torch.save(model if not len(device_list)>1  else model.module,model_save_path+'_epoch_%d'%(epoch_idx+1)+'.pth')
        elif max_acc < test_acc :
            max_acc = test_acc 
            torch.save(model if not len(device_list)>1  else model.module,model_save_path+'_epoch_%d'%(epoch_idx+1)+'.pth')
        
        if epoch_idx %25 == 0 :
            os.system("nvidia-smi")
        
        
        if epoch_idx >= args.start_scheduler_step:
            lr_scheduler.step()
        
                

def cluster(train_loader_cluster,cluster_number):
    model.eval()
    features_sum = []
    
    for i,(intensity,angle,target,cluster_target) in enumerate(train_loader_cluster):
        intensity = intensity.to(device).type(torch.float32)
        angle = angle.to(device).type(torch.float32)
        target = target.to(device)
        with torch.no_grad():
            # features,_,_ = model(intensity,angle)
            features = torch.rand((intensity.shape[0],128)).to(device)
            features = features.detach()
            features_sum.append(features)
    features = torch.cat(features_sum,dim=0)
    features = torch.split(features,args.cls_num_list.tolist(),dim=0)
    targets = [torch.tensor([]) for i in range(len(cluster_number))]
    
    for i in range(len(cluster_number)):
        if cluster_number[i]== 0 :
            continue
        elif cluster_number[i] > 1 :
            cluster_ids_x ,_ = kmeans(X=features[i],num_clusters=cluster_number[i],distance='cosine',tol=1e-3,iter_limit=40,device=device,tqdm_flag=False)
            targets[i] = cluster_ids_x
            # print("cluster",i,"cluster_ids_x.shape",cluster_ids_x.shape)
        else:
            targets[i]=torch.zeros(features[i].shape[0])
            
    cluster_number_sum = [sum(cluster_number[:i]) for i in range(len(cluster_number))]
    
    for i,k in enumerate(cluster_number_sum):
        if targets[i] != []:
            targets[i] = torch.add(targets[i],k)
            
    targets = torch.cat(targets,dim=0)
    targets = targets.numpy().tolist()
    return targets


def test():
    model.eval()
    ema.apply_shadow()
    total_acc = MulticlassAccuracy(args.class_num,average='micro').to(device)
    total_num = 0 
    avg_err = 0.0 
    batch_cnt = 0 
    with torch.no_grad():
        for i,(intensity,angle,labels230,_) in enumerate(train_loader_cluster):
            intensity,angle,labels230 = intensity.to(device).type(torch.float32),angle.to(device).type(torch.float32),labels230.to(device).type(torch.long)
            _,sp,_ = model(intensity,angle)
            err = lossfn(sp,labels230)
            avg_err += err.item()
            logits = sp.softmax(dim=1)
            total_num += labels230.shape[0]
            total_acc(logits,labels230)
            batch_cnt+=1 
    total_acc_val = total_acc.compute().cpu().item()
    print("test_acc:%s,test_err:%s"%(str(total_acc_val),str(avg_err/batch_cnt)))
    ema.restore()
    
    return total_acc_val,avg_err/batch_cnt
if __name__ == '__main__':
    train()
    writer.close()