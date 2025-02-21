import torch 
import math
import numpy as np 



class PositionEmbedding(torch.nn.Module):
    def __init__(self,seq_len,dimension) -> None:
        super(PositionEmbedding,self).__init__()
        self.position_embedding = torch.nn.Parameter(self.get_position_embedding(seq_len,dimension),requires_grad=False)

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

    def forward(self,q,k,v,att_mask):
        attention_score = torch.matmul(q,k.transpose(-1,-2))/ np.sqrt(self.d_k)
        # print(attention_score)
        if att_mask is not None :
            attention_score = attention_score.masked_fill(att_mask,-1e9)
        # print(attention_score)
        attn_weights = torch.nn.Softmax(dim=-1)(attention_score)
        # print(attn_weights)
        output = torch.matmul(attn_weights,v)
        return output,attention_score

class MultiHeadAttention(torch.nn.Module):
    def __init__(self,dimension):
        super(MultiHeadAttention,self).__init__()
        self.d_k = self.d_v = dimension 

        self.WQ = torch.nn.Linear(dimension,dimension)
        self.WK = torch.nn.Linear(dimension,dimension)
        self.WV = torch.nn.Linear(dimension,dimension)
        self.scaled_dot_product_attn = ScaledDotProductAttention(self.d_k)
        self.linear = torch.nn.Linear(self.d_v,dimension)

    def forward(self,X,att_mask=None):
        batch_size = X.shape[0]
        q = self.WQ(X).view(batch_size,-1,self.d_k)
        k = self.WK(X).view(batch_size,-1,self.d_k)
        v = self.WV(X).view(batch_size,-1,self.d_v)
        attn,atw= self.scaled_dot_product_attn(q,k,v,att_mask)
        attn = attn.contiguous().view(batch_size,-1,self.d_v)
        return self.linear(attn),atw

class FeedForwardNetwork(torch.nn.Module):
    def __init__(self,dimension,d_ff) -> None:
        super(FeedForwardNetwork,self).__init__()
        self.linear1 = torch.nn.Linear(dimension,d_ff)
        self.linear2 = torch.nn.Linear(d_ff,dimension)
        self.relu = torch.nn.LeakyReLU()

    def forward(self,X):
        return self.linear2(self.relu(self.linear1(X)))

class EncoderLayer(torch.nn.Module):
    def __init__(self,dimension,p_dropout,d_ff) -> None:
        super(EncoderLayer,self).__init__()
        self.mha = MultiHeadAttention(dimension)
        self.dropout1 = torch.nn.Dropout(p_dropout)
        self.layernorm1 = torch.nn.LayerNorm(dimension,eps=1e-6)
        
        self.ffn = FeedForwardNetwork(dimension,d_ff)

        self.dropout2 = torch.nn.Dropout(p_dropout)

        self.layernorm2 = torch.nn.LayerNorm(dimension,eps=1e-6)

    def forward(self,X,att_mask=None):
        # print("[EncoderLayer]X.shape:",X.shape)
        attn_output,atw= self.mha(X,att_mask)
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
        return ffn_output,atw

class TransformerEncoder(torch.nn.Module):
    def __init__(self,seq_len,dimension,n_layers,p_drop,d_ff) -> None:
        super(TransformerEncoder,self).__init__()
        self.positionEnbeding = PositionEmbedding(seq_len,dimension)
        self.encoder_layers = torch.nn.ModuleList([EncoderLayer(dimension,p_drop,d_ff) for _ in range(n_layers)])
    
    def forward(self,X,att_mask=None):
        # print("X.shape:",X.shape,"X:",X)
        outputs = self.positionEnbeding(X)
        # print("outputs.shape:",outputs.shape,"outputs:",outputs)
        for layer in self.encoder_layers :
            outputs,atw = layer(outputs,att_mask)
            # print("outputs.shape:",outputs.shape,"outputs:",outputs)
        return outputs

        
if __name__ == '__main__':
    device = torch.device('cuda:2')
    # position_embed = PositionEmbedding().to(device)
    X = torch.rand((2,3,4)).to(device)
    ma = ((torch.range(1,6).view(2,1,3))>4).to(device)
    print(X)
    encoder = TransformerEncoder(3,4,1,0.01,8).to(device)
    X = encoder(X,ma)
    print(X.shape)
