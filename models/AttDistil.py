import torch 
from models.TransformerEncoder import EncoderLayer,TransformerEncoder

SEQ_LEN=8500

class AttDistil(torch.nn.Module):
    def __init__(self,patch_len=10,embed_len=64,n_layers=8,n_heads=4,p_dropout=0,d_ff=256):
        super(AttDistil,self).__init__()
        self.embed = torch.nn.Embedding(SEQ_LEN,embed_len)
        self.encoder = TransformerEncoder(seq_len=SEQ_LEN//patch_len,dimension=patch_len*embed_len,n_layers=n_layers,n_heads=n_heads,p_drop=p_dropout,d_ff=d_ff)
        self.to_features = torch.nn.Linear(patch_len*embed_len,1024)
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
        # for encoder in self.encoders :
        #     x = encoder(x)
        x = self.encoder(x)
        x = self.to_features(x)
        features = x.mean(dim=1)
        cls = self.to_cls(features)
        return features,cls
    
        