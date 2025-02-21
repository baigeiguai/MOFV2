from torch.utils.data import DataLoader
from utils.dataset import XrdData
import os 
import torch 
import argparse 
from utils.logger import Log
from torchmetrics.classification import MulticlassAccuracy,MulticlassF1Score,MulticlassRecall
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
for k in models:
    models[k] = models[k].to(device)

logger.info('-'*15+'args'+'-'*15+'\n%s'%str(args))
logger.info('-'*15+'device'+'-'*15+'\n%s'%str(device))
logger.info('-'*15+'lossfn'+'-'*15+'\n%s'%str(lossfn))

def test():
    for k in tasks:
        models[k].eval()
        
    metrics = [0 for i in range(4)]  #total,lt_acc,cs_acc,gp_acc
    with torch.no_grad():
        for file in test_files:
            xrd_dataset = XrdData(file)
            dataloader = DataLoader(xrd_dataset,args.batch_size,num_workers=args.num_workers)
            for data in dataloader:
                intensity , angle,labels230,labels7,labels6,labels32 = data[0].type(torch.float).to(device),data[1].type(torch.float).to(device),data[2].to(device),data[4].to(device),data[5].to(device),data[6].to(device)
                rep = models['rep'](intensity,angle)
                out = models['sp'](rep)
                out = out.softmax(dim=-1)
                y_pred = torch.max(out,1).indices
                wrong_index = (y_pred != labels230)
                batch_metrics = calc_metrics(y_pred[wrong_index],labels230[wrong_index])
                for i in range(len(batch_metrics)):
                    metrics[i] +=batch_metrics[i]
    print("total_num:",metrics[0])
    print("lt_acc_num:",metrics[1],"rate:",metrics[1]/metrics[0])
    print("cs_acc_num:",metrics[2],"rate:",metrics[2]/metrics[0])
    print("pg_acc_num:",metrics[3],"rate:",metrics[3]/metrics[0])  
    
def calc_metrics(y_pred,y_hat):
    sp2lt = {13: 0, 8: 1, 147: 2, 0: 0, 1: 0, 143: 0, 61: 0, 18: 0, 166: 2, 14: 1, 165: 2, 146: 0, 4: 1, 154: 2, 224: 3, 208: 3, 138: 4, 122: 0, 78: 4, 160: 2, 17: 0, 3: 0, 5: 0, 60: 0, 210: 4, 145: 2, 159: 2, 7: 1, 33: 0, 155: 0, 142: 0, 25: 0, 216: 4, 196: 4, 81: 4, 177: 0, 19: 1, 168: 0, 11: 1, 125: 0, 74: 0, 84: 0, 228: 4, 23: 4, 80: 0, 15: 0, 214: 0, 220: 0, 151: 0, 28: 0, 42: 3, 86: 4, 96: 4, 71: 4, 112: 0, 12: 0, 9: 0, 70: 4, 200: 0, 47: 0, 68: 3, 64: 1, 20: 1, 174: 0, 136: 0, 93: 0, 55: 0, 39: 5, 49: 0, 213: 4, 119: 4, 173: 0, 171: 0, 52: 0, 16: 0, 186: 0, 48: 0, 85: 0, 153: 0, 126: 0, 89: 0, 115: 0, 207: 0, 137: 0, 22: 4, 123: 0, 229: 4, 141: 4, 194: 0, 199: 0, 181: 0, 117: 0, 161: 0, 190: 0, 2: 0, 176: 0, 218: 3, 225: 3, 29: 0, 149: 0, 77: 0, 98: 0, 27: 0, 69: 3, 180: 0, 90: 0, 124: 0, 53: 0, 140: 4, 87: 4, 97: 4, 156: 0, 6: 0, 41: 3, 54: 0, 204: 0, 75: 0, 184: 0, 189: 0, 95: 0, 226: 3, 209: 3, 56: 0, 58: 0, 50: 0, 167: 0, 211: 0, 76: 0, 30: 0, 32: 0, 101: 0, 206: 0, 170: 0, 157: 0, 104: 0, 113: 0, 118: 4, 183: 0, 182: 0, 158: 0, 185: 0, 88: 0, 179: 0, 187: 0, 82: 0, 129: 0, 175: 0, 65: 1, 192: 0, 37: 5, 193: 0, 172: 0, 127: 0, 108: 4, 150: 0, 169: 0, 102: 0, 162: 0, 164: 0, 152: 0, 144: 0, 128: 0, 223: 0, 105: 0, 109: 4, 163: 0, 178: 0, 73: 4, 188: 0, 197: 0, 222: 0, 21: 3, 10: 0, 198: 4, 46: 0, 31: 0, 227: 3, 121: 4, 120: 4, 103: 0, 72: 4, 35: 1, 131: 0, 221: 0, 100: 0, 191: 0, 114: 0, 203: 4, 201: 3, 111: 0, 26: 0, 83: 0, 135: 0, 34: 1, 215: 3, 59: 0, 217: 0, 148: 0, 106: 4, 45: 4, 130: 0, 139: 4, 195: 3, 134: 0, 99: 0, 44: 4, 91: 0, 107: 4, 24: 0, 57: 0, 202: 3, 36: 1, 212: 0, 132: 0, 43: 4, 133: 0, 92: 0, 110: 0, 94: 0, 219: 4, 62: 1, 51: 0, 116: 0, 205: 4, 79: 4,63: 1,66: 1,67: 1,40:5,38: 5}
    sp2cs = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 2, 16: 2, 17: 2, 18: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2, 30: 2, 31: 2, 32: 2, 33: 2, 34: 2, 35: 2, 36: 2, 37: 2, 38: 2, 39: 2, 40: 2, 41: 2, 42: 2, 43: 2, 44: 2, 45: 2, 46: 2, 47: 2, 48: 2, 49: 2, 50: 2, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2, 61: 2, 62: 2, 63: 2, 64: 2, 65: 2, 66: 2, 67: 2, 68: 2, 69: 2, 70: 2, 71: 2, 72: 2, 73: 2, 74: 3, 75: 3, 76: 3, 77: 3, 78: 3, 79: 3, 80: 3, 81: 3, 82: 3, 83: 3, 84: 3, 85: 3, 86: 3, 87: 3, 88: 3, 89: 3, 90: 3, 91: 3, 92: 3, 93: 3, 94: 3, 95: 3, 96: 3, 97: 3, 98: 3, 99: 3, 100: 3, 101: 3, 102: 3, 103: 3, 104: 3, 105: 3, 106: 3, 107: 3, 108: 3, 109: 3, 110: 3, 111: 3, 112: 3, 113: 3, 114: 3, 115: 3, 116: 3, 117: 3, 118: 3, 119: 3, 120: 3, 121: 3, 122: 3, 123: 3, 124: 3, 125: 3, 126: 3, 127: 3, 128: 3, 129: 3, 130: 3, 131: 3, 132: 3, 133: 3, 134: 3, 135: 3, 136: 3, 137: 3, 138: 3, 139: 3, 140: 3, 141: 3, 142: 4, 143: 4, 144: 4, 145: 4, 146: 4, 147: 4, 148: 4, 149: 4, 150: 4, 151: 4, 152: 4, 153: 4, 154: 4, 155: 4, 156: 4, 157: 4, 158: 4, 159: 4, 160: 4, 161: 4, 162: 4, 163: 4, 164: 4, 165: 4, 166: 4, 167: 5, 168: 5, 169: 5, 170: 5, 171: 5, 172: 5, 173: 5, 174: 5, 175: 5, 176: 5, 177: 5, 178: 5, 179: 5, 180: 5, 181: 5, 182: 5, 183: 5, 184: 5, 185: 5, 186: 5, 187: 5, 188: 5, 189: 5, 190: 5, 191: 5, 192: 5, 193: 5, 194: 6, 195: 6, 196: 6, 197: 6, 198: 6, 199: 6, 200: 6, 201: 6, 202: 6, 203: 6, 204: 6, 205: 6, 206: 6, 207: 6, 208: 6, 209: 6, 210: 6, 211: 6, 212: 6, 213: 6, 214: 6, 215: 6, 216: 6, 217: 6, 218: 6, 219: 6, 220: 6, 221: 6, 222: 6, 223: 6, 224: 6, 225: 6, 226: 6, 227: 6, 228: 6, 229: 6}
    sp2pg = {0:0,1:1,2:2,3:2,4:2,5:3,6:3,7:3,8:3,9:4,10:4,11:4,12:4,13:4,14:4,15:5,16:5,17:5,18:5,19:5,20:5,21:5,22:5,23:5,24:6,25:6,26:6,27:6,28:6,29:6,30:6,31:6,32:6,33:6,34:6,35:6,36:6,37:6,38:6,39:6,40:6,41:6,42:6,43:6,44:6,45:6,46:7,47:7,48:7,49:7,50:7,51:7,52:7,53:7,54:7,55:7,56:7,57:7,58:7,59:7,60:7,61:7,62:7,63:7,64:7,65:7,66:7,67:7,68:7,69:7,70:7,71:7,72:7,73:7,74:8,75:8,76:8,77:8,78:8,79:8,80:9,81:9,82:10,83:10,84:10,85:10,86:10,87:10,88:11,89:11,90:11,91:11,92:11,93:11,94:11,95:11,96:11,97:11,98:12,99:12,100:12,101:12,102:12,103:12,104:12,105:12,106:12,107:12,108:12,109:12,110:13,111:13,112:13,113:13,114:13,115:13,116:13,117:13,118:13,119:13,120:13,121:13,122:14,123:14,124:14,125:14,126:14,127:14,128:14,129:14,130:14,131:14,132:14,133:14,134:14,135:14,136:14,137:14,138:14,139:14,140:14,141:14,142:15,143:15,144:15,145:15,146:16,147:16,148:17,149:17,150:17,151:17,152:17,153:17,154:17,155:18,156:18,157:18,158:18,159:18,160:18,161:19,162:19,163:19,164:19,165:19,166:19,167:20,168:20,169:20,170:20,171:20,172:20,173:21,174:22,175:22,176:23,177:23,178:23,179:23,180:23,181:23,182:24,183:24,184:24,185:24,186:25,187:25,188:25,189:25,190:26,191:26,192:26,193:26,194:27,195:27,196:27,197:27,198:27,199:28,200:28,201:28,202:28,203:28,204:28,205:28,206:29,207:29,208:29,209:29,210:29,211:29,212:29,213:29,214:30,215:30,216:30,217:30,218:30,219:30,220:31,221:31,222:31,223:31,224:31,225:31,226:31,227:31,228:31,229:31}
    total  = y_pred.shape[0]
    lt_acc = sum([sp2lt[p]==sp2lt[h] for(p,h) in zip(y_pred.cpu().numpy(),y_hat.cpu().numpy())])
    cs_acc = sum([sp2cs[p]==sp2cs[h] for(p,h) in zip(y_pred.cpu().numpy(),y_hat.cpu().numpy())])
    gp_acc = sum([sp2pg[p]==sp2pg[h] for(p,h) in zip(y_pred.cpu().numpy(),y_hat.cpu().numpy())])
    return [total,lt_acc,cs_acc,gp_acc]   

                
if __name__ == '__main__':
    test()
    
    
    