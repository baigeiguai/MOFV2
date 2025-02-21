import torch

class ClsSp(torch.nn.Module):
    def __init__(self,features_dim=64):
        super(ClsSp,self).__init__()
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(features_dim,128),
            torch.nn.Linear(128,256),
            torch.nn.Linear(256,230),
        )
    
    def forward(self,x):
        return self.cls(x)
    

class ClsCs(torch.nn.Module):
    def __init__(self,features_dim=64):
        super(ClsCs,self).__init__()
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(features_dim,32),
            torch.nn.Linear(32,16),
            torch.nn.Linear(16,7),
        )
    
    def forward(self,x):
        return self.cls(x)

class ClsLt(torch.nn.Module):
    def __init__(self,features_dim=64):
        super(ClsLt,self).__init__()
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(features_dim,32),
            torch.nn.Linear(32,16),
            torch.nn.Linear(16,6),
        )
    
    def forward(self,x):
        return self.cls(x)
    
class ClsPg(torch.nn.Module):
    def __init__(self,features_dim=64):
        super(ClsPg,self).__init__()
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(features_dim,32),
            torch.nn.Linear(32,32),
        )
        
    def forward(self,x):
        return self.cls(x)

