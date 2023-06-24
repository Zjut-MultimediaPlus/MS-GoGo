import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math
from pdb import set_trace as stop
from .transformer_layers import SelfAttnLayer
from .backbone import Backbone,Backbone_split_res101,Backbone_split_res34,Backbone_split_res18,Backbone_split_res50,CNN
from .utils import custom_replace,weights_init
from .position_enc import PositionEmbeddingSine,positionalencoding2d

# Q2L zdxy+
class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)

        if self.bias:
            x = x + self.b
        return x

class MaxClassifier(nn.Module):
    def __init__(self, in_dim, label_num):
        super(MaxClassifier, self).__init__()
        #nn.BatchNorm1d(in_dim),
        self.ln1 = nn.Linear(in_features=in_dim,out_features=128,bias=True)
        self.activate = nn.LeakyReLU(128)
        self.bn = nn.BatchNorm1d(17)
        self.ln2 = nn.Linear(in_features=128, out_features=label_num, bias=True)

    def forward(self,seq):
        """
        Args:
            seq: (B,L,D) batch_size, length_of_seq, hidden_dim_of_seq
        Returns:
            result:(B,N) batch_size, label_num
        """
        out = self.ln1(seq)         # (B,17,17)
        out = self.activate(out)
        out = self.bn(out)
        out = self.ln2(out)

        results,_ = torch.max(out,dim=1)    # (B,17)
        return results

# zdxy+
class MLP(nn.Module):
    def __init__(self,hidden,num_class):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
        nn.BatchNorm1d(hidden),
        nn.Linear(in_features=hidden, out_features=512, bias=True),
        nn.Dropout(0.5), # delete
        nn.LeakyReLU(512),
        nn.BatchNorm1d(512),
        nn.Linear(in_features=512, out_features=num_class, bias=True),
        nn.Dropout(0.5),# delete
        )
        self.softmax = nn.Softmax()


    def forward(self,x):
        x = x.view(x.shape[0],x.shape[2])
        x = self.net(x)
        x = self.softmax(x)
        #print(x.shape)
        return x

class Predictor(nn.Module):
    def __init__(self,in_dim, hidden_dim):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,hidden_dim,bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,in_dim)
        )
    def forward(self,x):
        # x:(B,L,C)
        return self.net(x)

class SemanticTokenizer(nn.Module):
    def __init__(self, num_tokens, hidden_size, in_channel):
        super(SemanticTokenizer,self).__init__()
        self.L = num_tokens
        self.hidden_size = hidden_size

        self.conv = nn.Conv2d(in_channel, self.L, (1,1), (1,1))

        self.mlp  = nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=hidden_size//2),
            nn.Mish(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hidden_size//2,out_features=hidden_size),
            nn.Mish(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
        )

    def forward(self,x):
        b,c,h,w = x.shape
        att_map = self.conv(x)
        att_map = att_map.view([b,self.L,-1]).contiguous()
        att_map = torch.softmax(att_map, dim=-1)
        x = x.view([b,c,-1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', att_map,x)
        tokens = self.mlp(tokens)
        return tokens

class MS_GoGo(nn.Module):
    def __init__(self, args, num_labels, use_lmt, pos_emb=False, layers=3, heads=4, dropout=0.1, int_loss=0,
                 no_x_features=False):
        super(MS_GoGo, self).__init__()
        self.use_lmt = use_lmt
        self.use_month = args.use_month

        self.no_x_features = args.no_x_features  # (for no image features)
        self.bb_name = args.backbone


        hidden = 512
        L = 64 * 4
        self.backbone1 = CNN(3)
        self.backbone2 = CNN(8)
        self.backbone3 = CNN(2)
        self.backbone4 = CNN(3)

        # Label Embeddings
        self.label_input = torch.Tensor(np.arange(num_labels)).view(1, -1).long()
        self.label_lt = torch.nn.Embedding(num_labels, hidden, padding_idx=None)

        # zdxy+
        self.month_embedding = torch.nn.Embedding(12, hidden, padding_idx=None)
        self.month_embedding.apply(weights_init)
        self.loc_embedding = torch.nn.Embedding(11 * 26, hidden, padding_idx=None)
        self.loc_embedding.apply(weights_init)
        self.feature_embedding = torch.nn.Embedding(3, hidden, padding_idx=None)
        self.feature_embedding.apply(weights_init)
        self.channel_embedding = torch.nn.Embedding(4, hidden, padding_idx=None)
        self.channel_embedding.apply(weights_init)
        self.position_embedding = torch.nn.Embedding(100 + 17, hidden, padding_idx=None)
        self.position_embedding.apply(weights_init)

        # State Embeddings
        self.known_label_lt = torch.nn.Embedding(3, hidden, padding_idx=0)

        # Position Embeddings (for image features)
        self.use_pos_enc = pos_emb
        # zdxy original (hidden,18,18)
        self.position_encoding = positionalencoding2d(hidden, 8, 8).unsqueeze(0)
        self.use_geo_enc = args.geo_emb
        self.use_loc = args.use_loc

        '''
        if self.use_pos_enc:
            print(1)
            # self.position_encoding = PositionEmbeddingSine(int(hidden/2), normalize=True)
        '''

        # Transformer
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(hidden, heads, dropout) for _ in range(layers)])

        # Classifier
        # Output is of size num_labels because we want a separate classifier for each label
        self.output_linear = torch.nn.Linear(hidden, num_labels)
        self.max_classifier = MaxClassifier(hidden, num_labels)
        self.group_wise_linear = GroupWiseLinear(num_labels, hidden)
        # GroupWiseLinear(2*L, hidden)


        # zdxy+
        self.out_project = torch.nn.Linear(num_labels, 1)
        self.fc = GroupWiseLinear(num_class=num_labels, hidden_dim=hidden)
        self.mlp = MLP(hidden, num_labels)
        self.month_mlp = MLP(hidden, 12)
        self.loc_mlp = MLP(hidden, 286)

        # Other
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        self.label_lt.apply(weights_init)
        self.known_label_lt.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)

    def forward(self, images, mask, img_loc, month, loc_num):
        const_label_input = self.label_input.repeat(images.size(0), 1).cuda()
        init_label_embeddings = self.label_lt(const_label_input)

        shp = images.shape

        group1 = torch.cat((images[:, 0:2, :, :],
                            images[:, 3, :, :].view(shp[0], 1, shp[2], shp[3])), dim=1)
        group2 = torch.cat(
            (images[:, 2, :, :].view(shp[0], 1, shp[2], shp[3]), images[:, 4, :, :].view(shp[0], 1, shp[2], shp[3]),
             images[:, 6, :, :].view(shp[0], 1, shp[2], shp[3]),
             images[:, 10, :, :].view(shp[0], 1, shp[2], shp[3]),
             images[:, 12:16, :, :]), dim=1)
        group3 = torch.cat((images[:, 5, :, :].view(shp[0], 1, shp[2], shp[3]),
                            images[:, 11, :, :].view(shp[0], 1, shp[2], shp[3])), dim=1)
        group4 = images[:, 7:10, :, :]

        features1 = self.backbone1(group1)
        features2 = self.backbone2(group2)
        features3 = self.backbone3(group3)
        features4 = self.backbone4(group4)

        features1 = features1.view(features1.size(0), features1.size(1), -1).permute(0, 2, 1)
        features2 = features2.view(features2.size(0), features2.size(1), -1).permute(0, 2, 1)
        features3 = features3.view(features3.size(0), features3.size(1), -1).permute(0, 2, 1)
        features4 = features4.view(features4.size(0), features4.size(1), -1).permute(0, 2, 1)

        features = torch.cat((features1, features2, features3, features4), 1)

        month_out = None
        loc_out = None

        if self.use_month:
            month_embedding = self.month_embedding(month.long().cuda())
            # print(month)
            # month_out = self.month_mlp(month_embedding)
            month_out = self.month_mlp(self.month_embedding.weight.view(12, 1, -1))  # for month_loss computation
            # print('month:', self.month_embedding.weight.shape)
            month_embedding = month_embedding.view(month_embedding.shape[0], 1, -1)
            month_embedding = month_embedding.repeat(1, init_label_embeddings.shape[1], 1)
            init_label_embeddings += month_embedding

        if self.use_loc:
            loc_embedding = self.loc_embedding(loc_num.long().cuda())
            # loc_out = self.loc_mlp(loc_embedding)
            # print('loc:', self.loc_embedding.weight.shape)
            loc_out = self.loc_mlp(self.loc_embedding.weight.view(286, 1, -1))  # for loc_loss computation
            loc_embedding = loc_embedding.view(loc_embedding.shape[0], 1, -1)
            loc_embedding = loc_embedding.repeat(1, init_label_embeddings.shape[1], 1)
            init_label_embeddings += loc_embedding

        if self.use_lmt:
            # Convert mask values to positive integers for nn.Embedding
            label_feat_vec = custom_replace(mask, 0, 1, 2).long()
            # Get state embeddings
            state_embeddings = self.known_label_lt(label_feat_vec)
            # Add state embeddings to label embeddings
            init_label_embeddings += state_embeddings

        if self.no_x_features:
            embeddings = init_label_embeddings
        else:
            embeddings = torch.cat((features, init_label_embeddings), 1)

        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)

        attns = []
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]


        output = self.group_wise_linear(label_embeddings)
        return output, None, attns, month_out, loc_out