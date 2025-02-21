from torch.utils.data import DataLoader
from utils.dataset import XrdData
import os 
import torch 
import argparse 
from utils.logger import Log
from torchmetrics.classification import MulticlassAccuracy,MulticlassF1Score
from utils.convert import per_class_acc2hmt_acc

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
parser.add_argument("--balanced",type=bool,default=False)

args = parser.parse_args()

log = Log(__name__,'log/test','test_%s'%(args.test_name))
args.log_name = log.log_name
logger = log.get_log()

device = torch.device("cuda:%s"%args.device if torch.cuda.is_available() else 'cpu')
lossfn = torch.nn.CrossEntropyLoss()
file_paths = os.listdir(args.data_path)
train_files,test_files = [os.path.join(args.data_path,f)  for f in file_paths if f.startswith('train')],[os.path.join(args.data_path,f) for f in file_paths if f.startswith('test')]

model = torch.load(args.model_path,map_location=device)
logger.info('-'*15+'args'+'-'*15+'\n%s'%str(args))
logger.info('-'*15+'model'+'-'*15+'\n%s'%str(model))
logger.info('-'*15+'device'+'-'*15+'\n%s'%str(device))
logger.info('-'*15+'lossfn'+'-'*15+'\n%s'%str(lossfn))

def test():
    model.eval()
    h_acc,m_acc,t_acc,top_k_acc_val = None,None,None,None 
    total_acc = MulticlassAccuracy(num_classes=230,average='micro').to(device)
    crystal_system_accuracy = MulticlassAccuracy(num_classes=7,average='micro').to(device)
    lattice_type_accuracy = MulticlassAccuracy(num_classes=6,average='micro').to(device)
    point_group_accuracy = MulticlassAccuracy(num_classes=32,average='micro').to(device)
    f1_score = MulticlassF1Score(num_classes=230,top_k=1).to(device)
    top_k_acc = MulticlassAccuracy(num_classes=230,top_k=args.top_k,average='micro').to(device)

    acc_per_class = MulticlassAccuracy(num_classes=230,average=None).to(device)
    
    total_num = 0
    total_err = 0.0 
    batch_cnt = 0 
    
    with torch.no_grad():
        for file in (test_files if args.mode=="test" else train_files):
            xrd_dataset = XrdData(file)
            dataloader = DataLoader(xrd_dataset,batch_size=args.batch_size,num_workers=args.num_workers)
            for data in dataloader:
                intensity , angle,labels230,index,labels7,labels6,labels32= data[0].type(torch.float).to(device),data[1].type(torch.float).to(device),data[2].to(device),data[3].to(device),data[4].to(device),data[5].to(device),data[6].to(device)
                
                raw_logits,cs_raw_logits,lt_raw_logits,gp_raw_logits = model(intensity,angle)
                
                # raw_logits,hkl = out[0],out[1]
                # raw_logits = model(intensity,angle)
                err = lossfn(raw_logits,labels230)
                logits = raw_logits.softmax(dim=1)
                cs_logits = cs_raw_logits.softmax(dim=1)
                lt_logits = lt_raw_logits.softmax(dim=1)
                gp_logits = gp_raw_logits.softmax(dim=1)
                total_acc(logits,labels230)
                crystal_system_accuracy(cs_logits,labels7)
                # print(lt_logits.shape,labels6.shape,labels7.shape,lt_raw_logits.shape)
                lattice_type_accuracy(lt_logits,labels6)
                point_group_accuracy(gp_logits,labels32)
                f1_score(logits,labels230)
                top_k_acc(logits,labels230)
                acc_per_class(logits,labels230)
                
                total_err += err.item()
                total_num += labels230.shape[0]
                batch_cnt += 1
        total_acc_val = total_acc.compute().cpu().item()
        crystal_system_acc_val = crystal_system_accuracy.compute().cpu().item()
        lattice_type_acc_val = lattice_type_accuracy.compute().cpu().item()
        point_group_acc_val = point_group_accuracy.compute().cpu().item()
        f1_score = f1_score.compute().cpu().item()
        top_k_acc_val = top_k_acc.compute().cpu().item()
        per_class_acc = list(acc_per_class.compute().cpu().numpy())
        h_acc,m_acc,t_acc = per_class_acc2hmt_acc(per_class_acc,args.balanced) 
    
    logger.info('-'*15+'performance'+'-'*15+'\ntotal_num:%d\nerror:%f\ntotal_acc:%s\nf1_score:%s\ntop%d_acc:%s\nhead_acc:%s\nmedium_acc:%s\ntail_add:%s\ncrystal_system_acc:%s\nlattice_type_acc:%s\npoint_group_acc:%s\n'%(
        total_num,
        total_err/batch_cnt,
        total_acc_val,
        f1_score,
        args.top_k,
        top_k_acc_val,
        h_acc,
        m_acc,
        t_acc,
        crystal_system_acc_val,
        lattice_type_acc_val,
        point_group_acc_val,
    ))
    logger.info('-'*15+'per_class_acc'+'-'*15+'\n%s'%per_class_acc)

if __name__ == '__main__':
    test()
    
    
    