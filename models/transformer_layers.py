import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stop
from .utils import get_activation_fn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        # Custom method to return attn outputs. Otherwise same as nn.TransformerEncoderLayer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    def forward(self, src, src_mask= None, src_key_padding_mask = None):
        src2,attn = self.self_attn(src, src, src, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2) 
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src,attn


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        # Custom method to return attn outputs. Otherwise same as nn.TransformerEncoderLayer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    def forward(self, q,k,v, src_mask= None, src_key_padding_mask = None):
        out1,attn = self.self_attn(q, k, v, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)
        out1 = k + self.dropout1(out1)
        out1 = self.norm1(out1)
        out2 = self.linear2(self.dropout(self.activation(self.linear1(out1))))
        out2 = out2 + self.dropout2(out2)
        out2 = self.norm2(out2)
        return out2,attn
        

class SelfAttnLayer(nn.Module):
    def __init__(self, d_model, nhead = 4,dropout=0.1):
        super().__init__()
        self.transformer_layer = TransformerEncoderLayer(d_model, nhead, d_model*1, dropout=dropout, activation='relu')
        # self.transformer_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model, dropout=dropout, activation='gelu') 

    def forward(self,k,mask=None):
        attn = None
        k=k.transpose(0,1)  
        x,attn = self.transformer_layer(k,src_mask=mask)
        # x = self.transformer_layer(k,src_mask=mask)
        x=x.transpose(0,1)
        return x,attn

class QueryLayer(nn.Module):
    def __init__(self, d_model, nhead = 4,dropout=0.1):
        super().__init__()
        self.transformer_layer = TransformerDecoderLayer(d_model, nhead, d_model*1, dropout=dropout, activation='relu')
        # self.transformer_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model, dropout=dropout, activation='gelu')

    def forward(self,q,v,k,mask=None):
        attn = None
        k=k.transpose(0,1)
        v=v.transpose(0,1)
        x,attn = self.transformer_layer(q,k,v,src_mask=mask)
        # x = self.transformer_layer(k,src_mask=mask)
        x=x.transpose(0,1)
        return x,attn