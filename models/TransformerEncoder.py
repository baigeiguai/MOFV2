import torch 
import math
import numpy as np 



class PositionEmbedding(torch.nn.Module):
    def __init__(self,device,seq_len,dimension) -> None:
        super(PositionEmbedding,self).__init__()
        self.device = device
        self.position_embedding = self.get_position_embedding(seq_len,dimension).to(device)

    def forward(self,x):
        # print("x.shape",x.shape,"position_embedding.shape",self.position_embedding.shape)
        return x + self.position_embedding.repeat(x.shape[0],1,1)
    
    def get_position_embedding(self,input_len,dimension):
        pe = torch.zeros(input_len,dimension)
        position = torch.arange(0,input_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dimension, 2) *
                                -(math.log(10000.0) / dimension))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self,d_k) -> None:
        super(ScaledDotProductAttention,self).__init__()
        self.d_k = d_k 

    def forward(self,q,k,v):
        attention_score = torch.matmul(q,k.transpose(-1,-2))/ np.sqrt(self.d_k)
        attn_weights = torch.nn.Softmax(dim=-1)(attention_score)
        output = torch.matmul(attn_weights,v)
        return output

class MultiHeadAttention(torch.nn.Module):
    def __init__(self,dimension,n_heads):
        super(MultiHeadAttention,self).__init__()
        self.n_heads = n_heads 
        self.d_k = self.d_v = dimension//n_heads 

        self.WQ = torch.nn.Linear(dimension,dimension)
        self.WK = torch.nn.Linear(dimension,dimension)
        self.WV = torch.nn.Linear(dimension,dimension)
        self.scaled_dot_product_attn = ScaledDotProductAttention(self.d_k)
        self.linear = torch.nn.Linear(self.d_v*self.n_heads,dimension)

    def forward(self,X):
        batch_size = X.shape[0]
        q_heads = self.WQ(X).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        k_heads = self.WK(X).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        v_heads = self.WV(X).view(batch_size,-1,self.n_heads,self.d_v).transpose(1,2)
        attn = self.scaled_dot_product_attn(q_heads,k_heads,v_heads)
        attn = attn.transpose(1,2).contiguous().view(batch_size,-1,self.d_v*self.n_heads)
        return self.linear(attn)

class FeedForwardNetwork(torch.nn.Module):
    def __init__(self,dimension,d_ff) -> None:
        super(FeedForwardNetwork,self).__init__()
        self.linear1 = torch.nn.Linear(dimension,d_ff)
        self.linear2 = torch.nn.Linear(d_ff,dimension)
        self.relu = torch.nn.LeakyReLU()

    def forward(self,X):
        return self.linear2(self.relu(self.linear1(X)))

class EncoderLayer(torch.nn.Module):
    def __init__(self,dimension,n_heads,p_dropout,d_ff) -> None:
        super(EncoderLayer,self).__init__()
        self.mha = MultiHeadAttention(dimension,n_heads)
        self.dropout1 = torch.nn.Dropout(p_dropout)
        self.layernorm1 = torch.nn.LayerNorm(dimension,eps=1e-6)
        
        self.ffn = FeedForwardNetwork(dimension,d_ff)

        self.dropout2 = torch.nn.Dropout(p_dropout)

        self.layernorm2 = torch.nn.LayerNorm(dimension,eps=1e-6)

    def forward(self,X):
        # print("[EncoderLayer]X.shape:",X.shape)
        attn_output = self.mha(X)
        # print("[EncoderLayer]X.shape:",attn_output.shape)
        attn_output = self.dropout1(attn_output)
        # print("[EncoderLayer]X.shape:",attn_output.shape)
        attn_output = self.layernorm1(attn_output+X)
        # print("[EncoderLayer]X.shape:",attn_output.shape)
        ffn_output = self.ffn(attn_output)
        # print("[EncoderLayer]X.shape:",ffn_output.shape)
        ffn_output = self.dropout2(ffn_output)
        # print("[EncoderLayer]X.shape:",ffn_output.shape)
        ffn_output = self.layernorm2(ffn_output+attn_output)
        # print("[EncoderLayer]X.shape:",ffn_output.shape)
        return ffn_output

class TransformerEncoder(torch.nn.Module):
    def __init__(self,seq_len,dimension,n_layers,n_heads,p_drop,d_ff,device) -> None:
        super(TransformerEncoder,self).__init__()
        self.positionEnbeding = PositionEmbedding(device,seq_len,dimension)
        self.encoder_layers = torch.nn.ModuleList([EncoderLayer(dimension,n_heads,p_drop,d_ff) for _ in range(n_layers)])
    
    def forward(self,X):
        # print("X.shape:",X.shape,"X:",X)
        outputs = self.positionEnbeding(X)
        # print("outputs.shape:",outputs.shape,"outputs:",outputs)
        for layer in self.encoder_layers :
            outputs = layer(outputs)
            # print("outputs.shape:",outputs.shape,"outputs:",outputs)
        return outputs

        
if __name__ == '__main__':
    device = torch.device('cuda:2')
    # position_embed = PositionEmbedding().to(device)
    X = torch.rand((128,500,512)).to(device)
    # X = position_embed(X)
    encoder = TransformerEncoder(500,512,1,8,0.01,256,device).to(device)
    X = encoder(X)
