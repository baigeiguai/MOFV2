import numpy as np 
import torch 


WRAP0_NUM_PER_CLS = [5013, 9433, 75, 9261, 4497, 18, 2432, 175, 5626, 61, 2345, 903, 3526, 9425, 9425, 22, 38, 2148, 9325, 913, 31, 18, 126, 33, 2, 75, 17, 8, 4112, 70, 275, 89, 7400, 155, 5, 694, 58, 21, 0, 98, 0, 38, 1812, 36, 313, 72, 15, 37, 10, 54, 26, 590, 70, 256, 151, 1888, 496, 353, 133, 4498, 9396, 5199, 481, 0, 102, 66, 0, 0, 62, 588, 72, 214, 157, 143, 28, 472, 61, 421, 133, 126, 116, 705, 16, 47, 449, 666, 360, 1914, 3, 51, 45, 978, 4, 105, 46, 927, 47, 101, 1, 3, 3, 13, 17, 60, 1, 53, 11, 18, 23, 213, 2, 17, 138, 624, 3, 20, 32, 94, 16, 55, 87, 344, 78, 47, 30, 112, 20, 91, 88, 246, 70, 10, 25, 28, 49, 92, 56, 73, 115, 72, 126, 273, 113, 388, 385, 715, 617, 4443, 6, 54, 13, 483, 11, 381, 249, 2, 9, 51, 185, 128, 516, 29, 242, 48, 384, 312, 912, 12, 355, 346, 40, 30, 365, 16, 23, 397, 15, 146, 125, 40, 24, 96, 1, 7, 18, 47, 21, 7, 22, 88, 65, 59, 38, 181, 10, 57, 83, 292, 67, 12, 25, 23, 54, 64, 428, 70, 13, 2, 35, 30, 29, 24, 33, 14, 21, 21, 163, 70, 48, 146, 127, 77, 38, 16, 340, 41, 88, 71, 22, 58]
class CBLoss(torch.nn.Module):
    def __init__(self,num_per_cls=WRAP0_NUM_PER_CLS,beta=0.99,num_of_classes=230):
      super(CBLoss,self).__init__()
      num_per_cls = np.array(num_per_cls)
      effective_num = 1.0 - np.power(beta, num_per_cls+1)
      weights = (1.0 - beta) / np.array(effective_num)
      weights = weights / np.sum(weights) * num_of_classes
      weights = torch.nn.Parameter(torch.tensor(weights,dtype=torch.float),requires_grad=False)
      self.ce = torch.nn.CrossEntropyLoss(weight=weights)
    
    def forward(self,data,target):
       return self.ce(data,target)
     

if __name__ == '__main__':
  data = torch.rand((5,230))
  target = torch.arange(5).type(torch.long)
  lossfn = CBLoss(num_per_cls= [1.0 for i in range(230)])
  err = lossfn(data,target)
  print(err)
  