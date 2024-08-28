import torch 
from models.TransformerEncoder import EncoderLayer

SEQ_LEN=8500

class AttDistil(torch.nn.Module):
    def __init__(self,patch_len=10,embed_len=16,n_layers=24,n_heads=4,p_dropout=0,d_ff=256):
        super(AttDistil,self).__init__()
        self.embed = torch.nn.Embedding(SEQ_LEN,embed_len)
        self.encoders =  torch.nn.ModuleList([EncoderLayer(embed_len*patch_len,n_heads,p_dropout,d_ff) for _ in range(n_layers)])        
        self.to_features = torch.nn.Linear(SEQ_LEN*embed_len,1024)
        self.to_cls = torch.nn.Linear(1024,230)
        self.patch_len = patch_len
        self.embed_len = embed_len
        
        
    def forward(self,intensity,index):
        intensity = intensity.type(torch.float).view(intensity.shape[0],-1,1)/100
        index  = index.type(torch.long)
        # b*seq_len *embed_len
        x = self.embed(index)*intensity
        # b*seq_len/patch_len*patch_len*embed_len
        x = x.view(x.shape[0],SEQ_LEN//self.patch_len,-1)
        for encoder in self.encoders :
            x = encoder(x)
        x = x.view(x.shape[0],-1)
        features = self.to_features(x)
        cls = self.to_cls(features)
        return features,cls
    
        