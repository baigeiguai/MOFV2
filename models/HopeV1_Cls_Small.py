import torch

class ClsSp(torch.nn.Module):
    def __init__(self,features_dim=1056):
        super(ClsSp,self).__init__()
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(576,230),
        )
    
    def forward(self,x):
        return self.cls(x)
    

class ClsCs(torch.nn.Module):
    def __init__(self,features_dim=1056):
        super(ClsCs,self).__init__()
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(576,256),
            torch.nn.Linear(256,64),
            torch.nn.Linear(64,7),
        )
    
    def forward(self,x):
        return self.cls(x)

class ClsLt(torch.nn.Module):
    def __init__(self,features_dim=1056):
        super(ClsLt,self).__init__()
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(576,256),
            torch.nn.Linear(256,64),
            torch.nn.Linear(64,6),
        )
    
    def forward(self,x):
        return self.cls(x)
    
class ClsPg(torch.nn.Module):
    def __init__(self,features_dim=1056):
        super(ClsPg,self).__init__()
        self.cls = torch.nn.Sequential(
            torch.nn.Linear(576,256),
            torch.nn.Linear(256,32),
        )
        
    def forward(self,x):
        return self.cls(x)

