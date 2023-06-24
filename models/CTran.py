 
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

class CTranModel_16c(nn.Module):
    def __init__(self,args,num_labels,use_lmt,pos_emb=False,layers=3,heads=4,dropout=0.1,int_loss=0,no_x_features=False):
        super(CTranModel_16c, self).__init__()
        self.use_lmt = use_lmt
        self.use_month = args.use_month

        self.no_x_features = no_x_features # (for no image features)

        # ResNet backbone
        if args.backbone == 'res18':
            hidden = 512
            self.backbone = Backbone_split_res18(16)
        elif args.backbone == 'res34':
            hidden = 512
            self.backbone = Backbone_split_res34(16)
        elif args.backbone == 'res101':
            hidden = 2048
            self.backbone = Backbone_split_res101(16)
        elif args.backbone == 'cnn':
            hidden = 512
            self.backbone = CNN(16)

        #self.backbone = Backbone_split_res34(16)
        #hidden = 512 # this should match the backbone output feature size
        #zdxy
        self.hidden = hidden
        self.downsample = False
        if self.downsample:
            self.conv_downsample = torch.nn.Conv2d(hidden,hidden,(1,1))

        
        # Label Embeddings
        self.label_input = torch.Tensor(np.arange(num_labels)).view(1,-1).long()
        self.label_lt = torch.nn.Embedding(num_labels, hidden, padding_idx=None)

        # zdxy+
        self.month_embedding = torch.nn.Embedding(12, hidden, padding_idx=None)
        self.month_embedding.apply(weights_init)
        self.loc_embedding = torch.nn.Embedding(11*26, hidden, padding_idx=None)
        self.loc_embedding.apply(weights_init)
        self.feature_embedding = torch.nn.Embedding(2, hidden, padding_idx=None)
        self.feature_embedding.apply(weights_init)

        # State Embeddings
        self.known_label_lt = torch.nn.Embedding(3, hidden, padding_idx=0)

        # Position Embeddings (for image features)
        self.use_pos_enc = pos_emb
        #zdxy original (hidden,18,18)
        self.position_encoding = positionalencoding2d(hidden, 8, 8).unsqueeze(0)
        self.use_geo_enc = args.geo_emb
        self.use_loc = args.use_loc

        '''
        if self.use_pos_enc:
            print(1)
            # self.position_encoding = PositionEmbeddingSine(int(hidden/2), normalize=True)
        '''

        # Transformer
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(hidden,heads,dropout) for _ in range(layers)])

        # Classifier
        # Output is of size num_labels because we want a separate classifier for each label
        self.output_linear = torch.nn.Linear(hidden,num_labels)
        # zdxy+
        self.out_project = torch.nn.Linear(num_labels,1)
        self.fc = GroupWiseLinear(num_class=num_labels,hidden_dim=hidden)
        self.mlp = MLP(hidden,num_labels)
        self.month_mlp = MLP(hidden,12)
        self.loc_mlp = MLP(hidden,286)

        # Other
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        self.label_lt.apply(weights_init)
        self.known_label_lt.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)


    def forward(self,images,mask,img_loc,month,loc_num):
        const_label_input = self.label_input.repeat(images.size(0),1).cuda()
        init_label_embeddings = self.label_lt(const_label_input)

        features = self.backbone(images)
        
        if self.downsample:
            features = self.conv_downsample(features)
        if self.use_pos_enc:
            #zdxy
            '''
            pos_encoding = positionalencoding2d(d_model=self.hidden,
                                                width=features.shape[2],
                                                height=features.shape[3])
            '''
                #self.position_encoding(features,torch.zeros(features.size(0),18,18, dtype=torch.bool).cuda())
            #print(features.shape)
            #print(self.position_encoding.shape)
            features = features + self.position_encoding.cuda()

        if self.use_geo_enc:

            geo_encoding = positionalencoding2d(self.hidden,
                                                features.shape[2] * 6,
                                                features.shape[3] * 6)
            pos_encoding = torch.zeros_like(features)
            for idx in range(features.shape[0]):
                loc_x = int((img_loc[0][idx] - 1) * 1.6)
                loc_y = int((img_loc[1][idx] - 1) * 1.6)
                pos_encoding[idx, :, :, :] = geo_encoding[:, loc_y:loc_y + 8, loc_x:loc_x+ 8]

            features = features + pos_encoding

        features = features.view(features.size(0),features.size(1),-1).permute(0,2,1) 

        if self.use_lmt:
            # Convert mask values to positive integers for nn.Embedding
            label_feat_vec = custom_replace(mask,0,1,2).long()

            # Get state embeddings
            state_embeddings = self.known_label_lt(label_feat_vec)

            # Add state embeddings to label embeddings
            init_label_embeddings += state_embeddings

        month_out = None
        loc_out = None
        if self.use_month:
            month_embedding = self.month_embedding(month.long().cuda())
            #print(month)
            month_out = self.month_mlp(month_embedding)
            month_embedding =month_embedding.view(month_embedding.shape[0],1,-1)
            month_embedding = month_embedding.repeat(1,init_label_embeddings.shape[1],1)
            init_label_embeddings += month_embedding


        if self.use_loc:
            loc_embedding = self.loc_embedding(loc_num.long().cuda())
            loc_out = self.loc_mlp(loc_embedding)
            loc_embedding =loc_embedding.view(loc_embedding.shape[0],1,-1)
            loc_embedding = loc_embedding.repeat(1,init_label_embeddings.shape[1],1)
            init_label_embeddings += loc_embedding

        
        if self.no_x_features:
            embeddings = init_label_embeddings 
        else:
            # Concat image and label embeddings
            embeddings = torch.cat((features,init_label_embeddings),1)

            #zdxy+
            if self.use_month and self.use_loc:
                shp_t = month_embedding.shape
                #month_embedding = month_embedding.view(shp_t[0], 1, shp_t[1])
                #loc_embedding = loc_embedding.view(shp_t[0], 1, shp_t[1])
                #embeddings = torch.cat((features, month_embedding,
                #                        loc_embedding,init_label_embeddings), 1)

            elif self.use_month:
                #print(embeddings.shape,month_embedding.shape)
                shp_t = month_embedding.shape
                #month_embedding = month_embedding.view(shp_t[0],1,shp_t[1])
                #embeddings = torch.cat((features,month_embedding,init_label_embeddings),1)

            elif self.use_loc:
                shp_t = loc_embedding.shape
                #loc_embedding = loc_embedding.view(shp_t[0], 1, shp_t[1])
                #embeddings = torch.cat((features, loc_embedding,init_label_embeddings), 1)

            else:
                TODO:'hahah'
                #embeddings = torch.cat((features, init_label_embeddings), 1)


        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)        
        attns = []
        for layer in self.self_attn_layers:
            embeddings,attn = layer(embeddings,mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:,-init_label_embeddings.size(1):,:]
        #print(self.month_embedding.weight[0])
        output = self.output_linear(label_embeddings)
        diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0),1,1).cuda()

        #zdxy modified
        #can be another linear
        #output,_ = output.max(dim=2)
        output = (output*diag_mask).sum(-1)##
        #output = self.mlp(label_embeddings)

        #zdxy+
        #output = self.out_project(torch.relu(output)).view(output.shape[0],output.shape[1])
        #output = self.fc(label_embeddings)

        #print(self.month_embedding.weight[0:2, 0:4])
        return output,None,attns,month_out,loc_out

class CTranModel(nn.Module):
    def __init__(self, args, num_labels, use_lmt, pos_emb=False, layers=3, heads=4, dropout=0.1, int_loss=0,
                 no_x_features=False):
        super(CTranModel, self).__init__()
        self.use_lmt = use_lmt
        self.use_month = args.use_month

        self.no_x_features = no_x_features  # (for no image features)

        # ResNet backbone
        # ResNet backbone
        if args.backbone == 'res18':
            self.backbone = Backbone_split_res18(3)
            hidden = 512
        elif args.backbone == 'res34':
            self.backbone = Backbone_split_res34(3)
            hidden = 512
        elif args.backbone == 'res101':
            self.backbone = Backbone_split_res101(3)
            hidden = 2048
        elif args.backbone == 'cnn':
            self.backbone = CNN(3)
            hidden = 512
        #self.backbone = Backbone_split_res34(3)#Backbone()
          # this should match the backbone output feature size
        # zdxy
        self.hidden = hidden
        self.downsample = False
        if self.downsample:
            self.conv_downsample = torch.nn.Conv2d(hidden, hidden, (1, 1))

        # Label Embeddings
        self.label_input = torch.Tensor(np.arange(num_labels)).view(1, -1).long()
        self.label_lt = torch.nn.Embedding(num_labels, hidden, padding_idx=None)

        # zdxy+
        self.month_embedding = torch.nn.Embedding(12, hidden, padding_idx=None)
        self.month_embedding.apply(weights_init)
        self.loc_embedding = torch.nn.Embedding(11 * 26, hidden, padding_idx=None)
        self.loc_embedding.apply(weights_init)
        self.feature_embedding = torch.nn.Embedding(2, hidden, padding_idx=None)
        self.feature_embedding.apply(weights_init)

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
        self.max_mixer = MaxClassifier(hidden,num_labels)
        self.group_wise_linear = GroupWiseLinear(num_labels,hidden)
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
        #print(images.shape)
        features = self.backbone(images)
        #temp = cv2.merge((images[0,0,:,:].cpu().numpy()*255,images[0,1,:,:].cpu().numpy()*255,images[0,2,:,:].cpu().numpy()*255))
        #print(images[0])
        #cv2.imshow('hhh',temp)

        if self.downsample:
            features = self.conv_downsample(features)
        if self.use_pos_enc:
            # zdxy
            '''
            pos_encoding = positionalencoding2d(d_model=self.hidden,
                                                width=features.shape[2],
                                                height=features.shape[3])
            '''
            # self.position_encoding(features,torch.zeros(features.size(0),18,18, dtype=torch.bool).cuda())
            # print(features.shape)
            # print(self.position_encoding.shape)
            features = features + self.position_encoding.cuda()

        if self.use_geo_enc:

            geo_encoding = positionalencoding2d(self.hidden,
                                                features.shape[2] * 6,
                                                features.shape[3] * 6)
            pos_encoding = torch.zeros_like(features)
            for idx in range(features.shape[0]):
                loc_x = int((img_loc[0][idx] - 1) * 1.6)
                loc_y = int((img_loc[1][idx] - 1) * 1.6)
                pos_encoding[idx, :, :, :] = geo_encoding[:, loc_y:loc_y + 8, loc_x:loc_x + 8]

            features = features + pos_encoding

        features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)

        if self.use_lmt:
            # Convert mask values to positive integers for nn.Embedding
            label_feat_vec = custom_replace(mask, 0, 1, 2).long()

            # Get state embeddings
            state_embeddings = self.known_label_lt(label_feat_vec)

            # Add state embeddings to label embeddings
            init_label_embeddings += state_embeddings

        month_out = None
        loc_out = None
        if self.use_month:
            month_embedding = self.month_embedding(month.long().cuda())
            # print(month)
            #month_out = self.month_mlp(month_embedding)
            month_out = self.month_mlp(self.month_embedding.weight.view(12,1,-1))  # for month_loss computation
            #print('month:', self.month_embedding.weight.shape)
            month_embedding = month_embedding.view(month_embedding.shape[0], 1, -1)
            month_embedding = month_embedding.repeat(1, init_label_embeddings.shape[1], 1)
            init_label_embeddings += month_embedding

        if self.use_loc:
            loc_embedding = self.loc_embedding(loc_num.long().cuda())
            #loc_out = self.loc_mlp(loc_embedding)
            #print('loc:', self.loc_embedding.weight.shape)
            loc_out = self.loc_mlp(self.loc_embedding.weight.view(286,1,-1))  # for loc_loss computation
            loc_embedding = loc_embedding.view(loc_embedding.shape[0], 1, -1)
            loc_embedding = loc_embedding.repeat(1, init_label_embeddings.shape[1], 1)
            init_label_embeddings += loc_embedding

        if self.no_x_features:
            embeddings = init_label_embeddings
        else:
            # Concat image and label embeddings
            embeddings = torch.cat((features, init_label_embeddings), 1)

            # zdxy+
            if self.use_month and self.use_loc:
                shp_t = month_embedding.shape
                # month_embedding = month_embedding.view(shp_t[0], 1, shp_t[1])
                # loc_embedding = loc_embedding.view(shp_t[0], 1, shp_t[1])
                # embeddings = torch.cat((features, month_embedding,
                #                        loc_embedding,init_label_embeddings), 1)

            elif self.use_month:
                # print(embeddings.shape,month_embedding.shape)
                shp_t = month_embedding.shape
                # month_embedding = month_embedding.view(shp_t[0],1,shp_t[1])
                # embeddings = torch.cat((features,month_embedding,init_label_embeddings),1)

            elif self.use_loc:
                shp_t = loc_embedding.shape
                # loc_embedding = loc_embedding.view(shp_t[0], 1, shp_t[1])
                # embeddings = torch.cat((features, loc_embedding,init_label_embeddings), 1)

            else:
                TODO: 'hahah'
                # embeddings = torch.cat((features, init_label_embeddings), 1)

        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)
        attns = []
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]
        # print(self.month_embedding.weight[0])
        output = self.output_linear(label_embeddings)
        diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0), 1, 1).cuda()
        output = self.max_mixer(label_embeddings)
        output = self.group_wise_linear(label_embeddings)

        # zdxy modified
        # can be another linear
        # output,_ = output.max(dim=2)
        #output = (output * diag_mask).sum(-1)  ##
        # output = self.mlp(label_embeddings)

        # zdxy+
        # output = self.out_project(torch.relu(output)).view(output.shape[0],output.shape[1])
        # output = self.fc(label_embeddings)

        # print(self.month_embedding.weight[0:2, 0:4])
        return output, None, attns, month_out, loc_out


class CTranModel_split(nn.Module):
    def __init__(self, args, num_labels, use_lmt, pos_emb=False, layers=3, heads=4, dropout=0.1, int_loss=0,
                 no_x_features=False):
        super(CTranModel_split, self).__init__()
        self.use_lmt = use_lmt
        self.use_month = args.use_month

        self.no_x_features = no_x_features  # (for no image features)

        # ResNet backbone

        if args.backbone == 'res18':
            hidden = 512
            self.backbone1 = Backbone_split_res18(1)
            self.backbone2 = Backbone_split_res18(1)
            self.backbone3 = Backbone_split_res18(1)
        elif args.backbone == 'res34':
            hidden = 512
            self.backbone1 = Backbone_split_res34(1)
            self.backbone2 = Backbone_split_res34(1)
            self.backbone3 = Backbone_split_res34(1)
        elif args.backbone == 'res50':
            hidden = 2048
            self.backbone1 = Backbone_split_res50(1)
            self.backbone2 = Backbone_split_res50(1)
            self.backbone3 = Backbone_split_res50(1)
        elif args.backbone == 'res101':
            hidden = 2048
            self.backbone1 = Backbone_split_res101(1)
            self.backbone2 = Backbone_split_res101(1)
            self.backbone3 = Backbone_split_res101(1)
        elif args.backbone == 'cnn':
            hidden = 512
            self.backbone1 = CNN(1)
            self.backbone2 = CNN(1)
            self.backbone3 = CNN(1)
        # self.backbone_all = Backbone_split_res101(3)

        # hidden = 512  # this should match the backbone output feature size
        # zdxy
        self.hidden = hidden
        self.downsample = False
        if self.downsample:
            self.conv_downsample = torch.nn.Conv2d(hidden, hidden, (1, 1))

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
        self.channel_embedding = torch.nn.Embedding(3, hidden, padding_idx=None)
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

        channel1 = images[:, 0, :, :].view(images.shape[0], 1, images.shape[2], images.shape[3])
        channel2 = images[:, 1, :, :].view(images.shape[0], 1, images.shape[2], images.shape[3])
        channel3 = images[:, 2, :, :].view(images.shape[0], 1, images.shape[2], images.shape[3])

        shp = images.shape
        '''
        group1 = torch.cat((images[:, 0:2, :, :], images[:, 3, :, :].view(shp[0], 1, shp[2], shp[3])), dim=1)
        group2 = torch.cat((images[:, 2, :, :].view(shp[0], 1, shp[2], shp[3]),
                            images[:, 4, :, :].view(shp[0], 1, shp[2], shp[3]),
                            images[:, 6, :, :].view(shp[0], 1, shp[2], shp[3]),
                            images[:, 10, :, :].view(shp[0], 1, shp[2], shp[3]), images[:, 12:16, :, :]), dim=1)
        group3 = torch.cat((images[:, 5, :, :].view(shp[0], 1, shp[2], shp[3]), images[:, 7:10, :, :],
                            images[:, 11, :, :].view(shp[0], 1, shp[2], shp[3])), dim=1)
        '''
        features1 = self.backbone1(channel1)
        features2 = self.backbone2(channel2)
        features3 = self.backbone3(channel3)
        ''''''
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
        # features_all = self.backbone_all(images[:,1:4,:,:])

        '''
        if self.downsample:
            features = self.conv_downsample(features)
        '''
        '''
        if self.use_pos_enc:
            # zdxy

            pos_encoding = positionalencoding2d(d_model=self.hidden,
                                                width=features.shape[2],
                                                height=features.shape[3])

            # self.position_encoding(features,torch.zeros(features.size(0),18,18, dtype=torch.bool).cuda())
            # print(features.shape)
            # print(self.position_encoding.shape)
            features = features + self.position_encoding.cuda()


        if self.use_geo_enc:

            geo_encoding = positionalencoding2d(self.hidden,
                                                features.shape[2] * 6,
                                                features.shape[3] * 6)
            pos_encoding = torch.zeros_like(features)
            for idx in range(features.shape[0]):
                loc_x = int((img_loc[0][idx] - 1) * 1.6)
                loc_y = int((img_loc[1][idx] - 1) * 1.6)
                pos_encoding[idx, :, :, :] = geo_encoding[:, loc_y:loc_y + 8, loc_x:loc_x + 8]

            features = features + pos_encoding
        '''

        features1 = features1.view(features1.size(0), features1.size(1), -1).permute(0, 2, 1)
        features2 = features2.view(features2.size(0), features2.size(1), -1).permute(0, 2, 1)
        features3 = features3.view(features3.size(0), features3.size(1), -1).permute(0, 2, 1)
        # features_all = features_all.view(features_all.size(0), features_all.size(1), -1).permute(0, 2, 1)
        # print(features_all.shape)

        group1_embedding = self.channel_embedding(torch.LongTensor([0]).repeat(features1.shape[0]).cuda())
        group1_embedding = group1_embedding.view(group1_embedding.shape[0], 1, -1)
        group1_embedding = group1_embedding.repeat(1, features1.shape[1], 1)
        features1 = features1 + group1_embedding

        group2_embedding = self.channel_embedding(torch.LongTensor([1]).repeat(features1.shape[0]).cuda())
        group2_embedding = group2_embedding.view(group2_embedding.shape[0], 1, -1)
        group2_embedding = group2_embedding.repeat(1, features2.shape[1], 1)
        features2 = features2 + group2_embedding

        group3_embedding = self.channel_embedding(torch.LongTensor([1]).repeat(features1.shape[0]).cuda())
        group3_embedding = group3_embedding.view(group3_embedding.shape[0], 1, -1)
        group3_embedding = group3_embedding.repeat(1, features3.shape[1], 1)
        features3 = features3 + group3_embedding

        features = torch.cat((features1, features2, features3), 1)
        embeddings = torch.cat((features, init_label_embeddings), 1)
        # embeddings = features
        pos_embedding = self.position_embedding.weight
        # embeddings += pos_embedding

        # print(features1.shape)

        if self.use_lmt:
            # Convert mask values to positive integers for nn.Embedding
            label_feat_vec = custom_replace(mask, 0, 1, 2).long()

            # Get state embeddings
            state_embeddings = self.known_label_lt(label_feat_vec)

            # Add state embeddings to label embeddings
            init_label_embeddings += state_embeddings

        if self.use_month:
            month_embedding = self.month_embedding(month.long().cuda())
            # month_embedding =month_embedding.view(month_embedding.shape[0],1,-1)
            # month_embedding = month_embedding.repeat(1,init_label_embeddings.shape[1],1)
            # init_label_embeddings += month_embedding

        if self.use_loc:
            loc_embedding = self.loc_embedding(loc_num.long().cuda())
            # loc_embedding =loc_embedding.view(loc_embedding.shape[0],1,-1)
            # loc_embedding = loc_embedding.repeat(1,init_label_embeddings.shape[1],1)
            # init_label_embeddings += loc_embedding

        if self.no_x_features:
            embeddings = init_label_embeddings
        else:
            # Concat image and label embeddings
            embeddings = torch.cat((features, init_label_embeddings), 1)

            # zdxy+
            '''
            if self.use_month and self.use_loc:
                shp_t = month_embedding.shape
                month_embedding = month_embedding.view(shp_t[0], 1, shp_t[1])
                loc_embedding = loc_embedding.view(shp_t[0], 1, shp_t[1])
                embeddings = torch.cat((features, month_embedding,
                                        loc_embedding, init_label_embeddings), 1)

            elif self.use_month:
                # print(embeddings.shape,month_embedding.shape)
                shp_t = month_embedding.shape
                month_embedding = month_embedding.view(shp_t[0], 1, shp_t[1])
                embeddings = torch.cat((features, month_embedding, init_label_embeddings), 1)

            elif self.use_loc:
                shp_t = loc_embedding.shape
                loc_embedding = loc_embedding.view(shp_t[0], 1, shp_t[1])
                embeddings = torch.cat((features, loc_embedding, init_label_embeddings), 1)
            '''
        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)
        attns = []
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]
        output = self.output_linear(label_embeddings)
        output = self.max_classifier(label_embeddings)
        output = self.group_wise_linear(label_embeddings)
        diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0), 1, 1).cuda()

        # zdxy modified
        # can be another linear
        # output,_ = output.max(dim=2)
        #  $$ output = (output * diag_mask).sum(-1)  ##
        # output = self.mlp(label_embeddings)

        # zdxy+
        # output = self.out_project(torch.relu(output)).view(output.shape[0],output.shape[1])
        # output = self.fc(label_embeddings)

        return output, None, attns, month_out, loc_out

class CTranModel_split_16c(nn.Module):
    def __init__(self, args, num_labels, use_lmt, pos_emb=False, layers=3, heads=4, dropout=0.1, int_loss=0,
                 no_x_features=False):
        super(CTranModel_split_16c, self).__init__()
        self.use_lmt = use_lmt
        self.use_month = args.use_month

        self.no_x_features = args.no_x_features  # (for no image features)

        # ResNet backbone
        '''
        if args.backbone == 'res18':
            hidden = 512
            self.backbone1 = Backbone_split_res18(3)
            self.backbone2 = Backbone_split_res18(8)
            self.backbone3 = Backbone_split_res18(5)
        elif args.backbone == 'res34':
            hidden = 512
            self.backbone1 = Backbone_split_res34(3)
            self.backbone2 = Backbone_split_res34(8)
            self.backbone3 = Backbone_split_res34(5)
        elif args.backbone == 'res50':
            hidden = 2048
            self.backbone1 = Backbone_split_res50(3)
            self.backbone2 = Backbone_split_res50(8)
            self.backbone3 = Backbone_split_res50(5)
        elif args.backbone == 'res101':
            hidden = 2048
            self.backbone1 = Backbone_split_res101(3)
            self.backbone2 = Backbone_split_res101(8)
            self.backbone3 = Backbone_split_res101(5)
        elif args.backbone == 'cnn':
            hidden = 512
            self.backbone1 = CNN(3)
            self.backbone2 = CNN(8)
            self.backbone3 = CNN(5)
        '''

        if args.backbone == 'res18':
            hidden = 512
            self.backbone1 = Backbone_split_res18(3)
            self.backbone2 = Backbone_split_res18(8)
            self.backbone3 = Backbone_split_res18(5)
        elif args.backbone == 'res34':
            hidden = 512
            self.backbone1 = Backbone_split_res34(3)
            self.backbone2 = Backbone_split_res34(8)
            self.backbone3 = Backbone_split_res34(5)
        elif args.backbone == 'res50':
            hidden = 2048
            self.backbone1 = Backbone_split_res50(3)
            self.backbone2 = Backbone_split_res50(8)
            self.backbone3 = Backbone_split_res50(5)
        elif args.backbone == 'res101':
            hidden = 2048
            self.backbone1 = Backbone_split_res101(3)
            self.backbone2 = Backbone_split_res101(8)
            self.backbone3 = Backbone_split_res101(5)
            self.backbone4 = Backbone_split_res101(3)
        elif args.backbone == 'cnn':
            hidden = 512
            self.backbone1 = CNN(3)
            self.backbone2 = CNN(8)
            self.backbone3 = CNN(2)
            self.backbone4 = CNN(3)
        #self.backbone_all = Backbone_split_res101(3)

        #hidden = 512  # this should match the backbone output feature size
        # zdxy
        self.hidden = hidden
        self.downsample = False
        if self.downsample:
            self.conv_downsample = torch.nn.Conv2d(hidden, hidden, (1, 1))

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
        self.position_embedding = torch.nn.Embedding(100+17, hidden, padding_idx=None)
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
        self.max_classifier = MaxClassifier(hidden,num_labels)
        self.group_wise_linear = GroupWiseLinear(num_labels,hidden)
        # zdxy+
        self.out_project = torch.nn.Linear(num_labels, 1)
        self.fc = GroupWiseLinear(num_class=num_labels, hidden_dim=hidden)
        self.mlp = MLP(hidden, num_labels)
        self.month_mlp = MLP(hidden,12)
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

        channel1 = images[:, 0, :, :].view(images.shape[0],1,images.shape[2],images.shape[3])
        channel2 = images[:, 1, :, :].view(images.shape[0],1,images.shape[2],images.shape[3])
        channel3 = images[:, 2, :, :].view(images.shape[0],1,images.shape[2],images.shape[3])

        shp = images.shape
        group1 = torch.cat((images[:, 0:2, :, :],
                            images[:, 3, :, :].view(shp[0],1,shp[2],shp[3])),dim=1)
        group2 = torch.cat((images[:, 2, :, :].view(shp[0],1,shp[2],shp[3]),images[:, 4, :, :].view(shp[0],1,shp[2],shp[3]),
                            images[:, 6, :, :].view(shp[0],1,shp[2],shp[3]),
                            images[:, 10, :, :].view(shp[0],1,shp[2],shp[3]),
                            images[:, 12:16, :, :]), dim=1)
        group3 = torch.cat((images[:, 5, :, :].view(shp[0],1,shp[2],shp[3]),
                            images[:, 11, :, :].view(shp[0],1,shp[2],shp[3])), dim=1)
        group4 = images[:, 7:10, :, :]

        features1 = self.backbone1(group1)
        features2 = self.backbone2(group2)
        features3 = self.backbone3(group3)
        features4 = self.backbone4(group4)


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
        #features_all = self.backbone_all(images[:,1:4,:,:])

        '''
        if self.downsample:
            features = self.conv_downsample(features)
        '''
        '''
        if self.use_pos_enc:
            # zdxy
            
            pos_encoding = positionalencoding2d(d_model=self.hidden,
                                                width=features.shape[2],
                                                height=features.shape[3])
            
            # self.position_encoding(features,torch.zeros(features.size(0),18,18, dtype=torch.bool).cuda())
            # print(features.shape)
            # print(self.position_encoding.shape)
            features = features + self.position_encoding.cuda()
        

        if self.use_geo_enc:

            geo_encoding = positionalencoding2d(self.hidden,
                                                features.shape[2] * 6,
                                                features.shape[3] * 6)
            pos_encoding = torch.zeros_like(features)
            for idx in range(features.shape[0]):
                loc_x = int((img_loc[0][idx] - 1) * 1.6)
                loc_y = int((img_loc[1][idx] - 1) * 1.6)
                pos_encoding[idx, :, :, :] = geo_encoding[:, loc_y:loc_y + 8, loc_x:loc_x + 8]

            features = features + pos_encoding
        '''

        features1 = features1.view(features1.size(0), features1.size(1), -1).permute(0, 2, 1)
        features2 = features2.view(features2.size(0), features2.size(1), -1).permute(0, 2, 1)
        features3 = features3.view(features3.size(0), features3.size(1), -1).permute(0, 2, 1)
        features4 = features4.view(features3.size(0), features4.size(1), -1).permute(0, 2, 1)
        #features_all = features_all.view(features_all.size(0), features_all.size(1), -1).permute(0, 2, 1)
        #print(features_all.shape)


        group1_embedding = self.channel_embedding(torch.LongTensor([0]).repeat(features1.shape[0]).cuda())
        group1_embedding = group1_embedding.view(group1_embedding.shape[0],1,-1)
        group1_embedding = group1_embedding.repeat(1,features1.shape[1],1)
        #features1 = features1 + group1_embedding

        group2_embedding = self.channel_embedding(torch.LongTensor([1]).repeat(features1.shape[0]).cuda())
        group2_embedding = group2_embedding.view(group2_embedding.shape[0], 1, -1)
        group2_embedding = group2_embedding.repeat(1, features2.shape[1], 1)
        #features2 = features2 + group2_embedding

        group3_embedding = self.channel_embedding(torch.LongTensor([2]).repeat(features1.shape[0]).cuda())
        group3_embedding = group3_embedding.view(group3_embedding.shape[0], 1, -1)
        group3_embedding = group3_embedding.repeat(1, features3.shape[1], 1)
        #features3 = features3 + group3_embedding

        group4_embedding = self.channel_embedding(torch.LongTensor([3]).repeat(features1.shape[0]).cuda())
        group4_embedding = group4_embedding.view(group3_embedding.shape[0], 1, -1)
        group4_embedding = group4_embedding.repeat(1, features3.shape[1], 1)
        #features4 = features4 + group4_embedding





        features = torch.cat((features1, features2, features3,features4), 1)
        embeddings = torch.cat((features,init_label_embeddings), 1)
        #embeddings = features
        pos_embedding = self.position_embedding.weight
        #embeddings += pos_embedding

        #print(features1.shape)

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
            # Concat image and label embeddings
            embeddings = torch.cat((features,init_label_embeddings),1)

            # zdxy+
            '''
            if self.use_month and self.use_loc:
                shp_t = month_embedding.shape
                month_embedding = month_embedding.view(shp_t[0], 1, shp_t[1])
                loc_embedding = loc_embedding.view(shp_t[0], 1, shp_t[1])
                embeddings = torch.cat((features, month_embedding,
                                        loc_embedding, init_label_embeddings), 1)

            elif self.use_month:
                # print(embeddings.shape,month_embedding.shape)
                shp_t = month_embedding.shape
                month_embedding = month_embedding.view(shp_t[0], 1, shp_t[1])
                embeddings = torch.cat((features, month_embedding, init_label_embeddings), 1)

            elif self.use_loc:
                shp_t = loc_embedding.shape
                loc_embedding = loc_embedding.view(shp_t[0], 1, shp_t[1])
                embeddings = torch.cat((features, loc_embedding, init_label_embeddings), 1)
            '''
        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)
        attns = []
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]
        output = self.output_linear(label_embeddings)
        #output = self.max_classifier(label_embeddings)
        output = self.group_wise_linear(label_embeddings)
        diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0), 1, 1).cuda()

        # zdxy modified
        # can be another linear
        # output,_ = output.max(dim=2)
        #output = (output * diag_mask).sum(-1)  ##
        # output = self.mlp(label_embeddings)

        # zdxy+
        # output = self.out_project(torch.relu(output)).view(output.shape[0],output.shape[1])
        # output = self.fc(label_embeddings)

        return output, None, attns, month_out, loc_out

class Multichannel3(nn.Module):
    def __init__(self, args, num_labels, use_lmt, pos_emb=False, layers=3, heads=4, dropout=0.1, int_loss=0,
                 no_x_features=False):
        super(Multichannel3, self).__init__()
        self.use_lmt = use_lmt
        self.use_month = args.use_month

        self.no_x_features = no_x_features  # (for no image features)

        self.conv_set = []
        for i in range(3):
            self.conv_set.append(torch.nn.Conv2d(1,8,(225,225),(2,2)))

        hidden = 2048
        self.hidden = hidden
        self.downsample = False
        if self.downsample:
            self.conv_downsample = torch.nn.Conv2d(hidden, hidden, (1, 1))

        # Label Embeddings
        self.label_input = torch.Tensor(np.arange(num_labels)).view(1, -1).long()
        self.label_lt = torch.nn.Embedding(num_labels, hidden, padding_idx=None)

        # zdxy+
        self.month_embedding = torch.nn.Embedding(12, hidden, padding_idx=None)
        self.month_embedding.apply(weights_init)
        self.loc_embedding = torch.nn.Embedding(11 * 26, hidden, padding_idx=None)
        self.loc_embedding.apply(weights_init)
        self.feature_embedding = torch.nn.Embedding(2, hidden, padding_idx=None)
        self.feature_embedding.apply(weights_init)

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
        #print(images.shape)
        features = []
        for i in range(3):
            feat_2d = self.conv_set[i].cuda()(images[:,i,:,:].view(images.shape[0],1,images.shape[2],images.shape[3])) #[N, 8, 16, 16]
            feat_2d = feat_2d.view(feat_2d.size(0), feat_2d.size(1), -1).permute(0, 2, 1)#[N, 8, 256]
            feat_1d = feat_2d.reshape(feat_2d.size(0),1,-1)
            features.append(feat_1d)

        features = torch.cat(features,dim=1)
        if self.downsample:
            features = self.conv_downsample(features)
        if self.use_pos_enc:
            # zdxy
            '''
            pos_encoding = positionalencoding2d(d_model=self.hidden,
                                                width=features.shape[2],
                                                height=features.shape[3])
            '''
            # self.position_encoding(features,torch.zeros(features.size(0),18,18, dtype=torch.bool).cuda())
            # print(features.shape)
            # print(self.position_encoding.shape)
            features = features + self.position_encoding.cuda()

        if self.use_geo_enc:

            geo_encoding = positionalencoding2d(self.hidden,
                                                features.shape[2] * 6,
                                                features.shape[3] * 6)
            pos_encoding = torch.zeros_like(features)
            for idx in range(features.shape[0]):
                loc_x = int((img_loc[0][idx] - 1) * 1.6)
                loc_y = int((img_loc[1][idx] - 1) * 1.6)
                pos_encoding[idx, :, :, :] = geo_encoding[:, loc_y:loc_y + 8, loc_x:loc_x + 8]

            features = features + pos_encoding


        if self.use_lmt:
            # Convert mask values to positive integers for nn.Embedding
            label_feat_vec = custom_replace(mask, 0, 1, 2).long()

            # Get state embeddings
            state_embeddings = self.known_label_lt(label_feat_vec)

            # Add state embeddings to label embeddings
            init_label_embeddings += state_embeddings

        month_out = None
        loc_out = None
        if self.use_month:
            month_embedding = self.month_embedding(month.long().cuda())
            # print(month)
            month_out = self.month_mlp(month_embedding)
            month_embedding = month_embedding.view(month_embedding.shape[0], 1, -1)
            month_embedding = month_embedding.repeat(1, init_label_embeddings.shape[1], 1)
            init_label_embeddings += month_embedding

        if self.use_loc:
            loc_embedding = self.loc_embedding(loc_num.long().cuda())
            loc_out = self.loc_mlp(loc_embedding)
            loc_embedding = loc_embedding.view(loc_embedding.shape[0], 1, -1)
            loc_embedding = loc_embedding.repeat(1, init_label_embeddings.shape[1], 1)
            init_label_embeddings += loc_embedding

        if self.no_x_features:
            embeddings = init_label_embeddings
        else:
            # Concat image and label embeddings
            #print(features.shape,init_label_embeddings.shape)
            embeddings = torch.cat((features, init_label_embeddings), 1)

            # zdxy+
            if self.use_month and self.use_loc:
                shp_t = month_embedding.shape
                # month_embedding = month_embedding.view(shp_t[0], 1, shp_t[1])
                # loc_embedding = loc_embedding.view(shp_t[0], 1, shp_t[1])
                # embeddings = torch.cat((features, month_embedding,
                #                        loc_embedding,init_label_embeddings), 1)

            elif self.use_month:
                # print(embeddings.shape,month_embedding.shape)
                shp_t = month_embedding.shape
                # month_embedding = month_embedding.view(shp_t[0],1,shp_t[1])
                # embeddings = torch.cat((features,month_embedding,init_label_embeddings),1)

            elif self.use_loc:
                shp_t = loc_embedding.shape
                # loc_embedding = loc_embedding.view(shp_t[0], 1, shp_t[1])
                # embeddings = torch.cat((features, loc_embedding,init_label_embeddings), 1)

            else:
                TODO: 'hahah'
                # embeddings = torch.cat((features, init_label_embeddings), 1)

        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)
        attns = []
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]
        # print(self.month_embedding.weight[0])
        output = self.output_linear(label_embeddings)
        diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0), 1, 1).cuda()

        # zdxy modified
        # can be another linear
        # output,_ = output.max(dim=2)
        output = (output * diag_mask).sum(-1)  ##
        # output = self.mlp(label_embeddings)

        # zdxy+
        # output = self.out_project(torch.relu(output)).view(output.shape[0],output.shape[1])
        # output = self.fc(label_embeddings)

        # print(self.month_embedding.weight[0:2, 0:4])
        return output, None, attns, month_out, loc_out
'''
class Multichannel16(nn.Module):
    def __init__(self, args, num_labels, use_lmt, pos_emb=False, layers=3, heads=4, dropout=0.1, int_loss=0,
                 no_x_features=False):
        super(Multichannel16, self).__init__()
        self.use_lmt = use_lmt
        self.use_month = args.use_month

        self.no_x_features = no_x_features  # (for no image features)

        self.cnn_set = []
        for i in range(16):
            self.cnn_set.append(CNN(1))

        hidden = 512
        self.hidden = hidden
        self.downsample = False
        if self.downsample:
            self.conv_downsample = torch.nn.Conv2d(hidden, hidden, (1, 1))

        # Label Embeddings
        self.label_input = torch.Tensor(np.arange(num_labels)).view(1, -1).long()
        self.label_lt = torch.nn.Embedding(num_labels, hidden, padding_idx=None)

        # zdxy+
        self.month_embedding = torch.nn.Embedding(12, hidden, padding_idx=None)
        self.month_embedding.apply(weights_init)
        self.loc_embedding = torch.nn.Embedding(11 * 26, hidden, padding_idx=None)
        self.loc_embedding.apply(weights_init)
        self.feature_embedding = torch.nn.Embedding(2, hidden, padding_idx=None)
        self.feature_embedding.apply(weights_init)

        # State Embeddings
        self.known_label_lt = torch.nn.Embedding(3, hidden, padding_idx=0)

        # Position Embeddings (for image features)
        self.use_pos_enc = pos_emb
        # zdxy original (hidden,18,18)
        self.position_encoding = positionalencoding2d(hidden, 8, 8).unsqueeze(0)
        self.use_geo_enc = args.geo_emb
        self.use_loc = args.use_loc

        '''


class Together(nn.Module):
    def __init__(self, args, num_labels, use_lmt, pos_emb=False, layers=3, heads=4, dropout=0.1, int_loss=0,
                 no_x_features=False):
        super(Together, self).__init__()
        self.use_lmt = use_lmt
        self.use_month = args.use_month

        self.no_x_features = no_x_features  # (for no image features)

        # ResNet backbone
        self.backbone = CNN(16)
        # self.backbone_all = Backbone_split_res101(3)

        # hidden = 512  # this should match the backbone output feature size
        # zdxy
        hidden = 512
        self.hidden = hidden
        self.downsample = False
        if self.downsample:
            self.conv_downsample = torch.nn.Conv2d(hidden, hidden, (1, 1))

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
        self.channel_embedding = torch.nn.Embedding(16, hidden, padding_idx=None)
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

        features = self.backbone(images)
        features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)

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
        # features_all = self.backbone_all(images[:,1:4,:,:])

        '''
        if self.downsample:
            features = self.conv_downsample(features)
        '''
        '''
        if self.use_pos_enc:
            # zdxy

            pos_encoding = positionalencoding2d(d_model=self.hidden,
                                                width=features.shape[2],
                                                height=features.shape[3])

            # self.position_encoding(features,torch.zeros(features.size(0),18,18, dtype=torch.bool).cuda())
            # print(features.shape)
            # print(self.position_encoding.shape)
            features = features + self.position_encoding.cuda()


        if self.use_geo_enc:

            geo_encoding = positionalencoding2d(self.hidden,
                                                features.shape[2] * 6,
                                                features.shape[3] * 6)
            pos_encoding = torch.zeros_like(features)
            for idx in range(features.shape[0]):
                loc_x = int((img_loc[0][idx] - 1) * 1.6)
                loc_y = int((img_loc[1][idx] - 1) * 1.6)
                pos_encoding[idx, :, :, :] = geo_encoding[:, loc_y:loc_y + 8, loc_x:loc_x + 8]

            features = features + pos_encoding
        '''
        embeddings = torch.cat((features, init_label_embeddings,), 1)
        # embeddings = features
        pos_embedding = self.position_embedding.weight
        # embeddings += pos_embedding

        # print(features1.shape)

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
            # Concat image and label embeddings
            embeddings = torch.cat((features, init_label_embeddings), 1)

            # zdxy+
            '''
            if self.use_month and self.use_loc:
                shp_t = month_embedding.shape
                month_embedding = month_embedding.view(shp_t[0], 1, shp_t[1])
                loc_embedding = loc_embedding.view(shp_t[0], 1, shp_t[1])
                embeddings = torch.cat((features, month_embedding,
                                        loc_embedding, init_label_embeddings), 1)

            elif self.use_month:
                # print(embeddings.shape,month_embedding.shape)
                shp_t = month_embedding.shape
                month_embedding = month_embedding.view(shp_t[0], 1, shp_t[1])
                embeddings = torch.cat((features, month_embedding, init_label_embeddings), 1)

            elif self.use_loc:
                shp_t = loc_embedding.shape
                loc_embedding = loc_embedding.view(shp_t[0], 1, shp_t[1])
                embeddings = torch.cat((features, loc_embedding, init_label_embeddings), 1)
            '''
        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)
        attns = []
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]
        # output = self.output_linear(label_embeddings)
        # output = self.max_classifier(label_embeddings)
        output = self.group_wise_linear(label_embeddings)
        diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0), 1, 1).cuda()

        # zdxy modified
        # can be another linear
        # output,_ = output.max(dim=2)
        #  $$ output = (output * diag_mask).sum(-1)  ##
        # output = self.mlp(label_embeddings)

        # zdxy+
        # output = self.out_project(torch.relu(output)).view(output.shape[0],output.shape[1])
        # output = self.fc(label_embeddings)

        return output, None, attns, month_out, loc_out

class Together_3(nn.Module):
    def __init__(self, args, num_labels, use_lmt, pos_emb=False, layers=3, heads=4, dropout=0.1, int_loss=0,
                 no_x_features=False):
        super(Together_3, self).__init__()
        self.use_lmt = use_lmt
        self.use_month = args.use_month

        self.no_x_features = no_x_features  # (for no image features)

        # ResNet backbone
        self.backbone = CNN(3)
        # self.backbone_all = Backbone_split_res101(3)

        # hidden = 512  # this should match the backbone output feature size
        # zdxy
        hidden = 512
        self.hidden = hidden
        self.downsample = False
        if self.downsample:
            self.conv_downsample = torch.nn.Conv2d(hidden, hidden, (1, 1))

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
        self.channel_embedding = torch.nn.Embedding(16, hidden, padding_idx=None)
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

        features = self.backbone(images)
        features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)

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
        # features_all = self.backbone_all(images[:,1:4,:,:])

        '''
        if self.downsample:
            features = self.conv_downsample(features)
        '''
        '''
        if self.use_pos_enc:
            # zdxy

            pos_encoding = positionalencoding2d(d_model=self.hidden,
                                                width=features.shape[2],
                                                height=features.shape[3])

            # self.position_encoding(features,torch.zeros(features.size(0),18,18, dtype=torch.bool).cuda())
            # print(features.shape)
            # print(self.position_encoding.shape)
            features = features + self.position_encoding.cuda()


        if self.use_geo_enc:

            geo_encoding = positionalencoding2d(self.hidden,
                                                features.shape[2] * 6,
                                                features.shape[3] * 6)
            pos_encoding = torch.zeros_like(features)
            for idx in range(features.shape[0]):
                loc_x = int((img_loc[0][idx] - 1) * 1.6)
                loc_y = int((img_loc[1][idx] - 1) * 1.6)
                pos_encoding[idx, :, :, :] = geo_encoding[:, loc_y:loc_y + 8, loc_x:loc_x + 8]

            features = features + pos_encoding
        '''
        embeddings = torch.cat((features, init_label_embeddings,), 1)
        # embeddings = features
        pos_embedding = self.position_embedding.weight
        # embeddings += pos_embedding

        # print(features1.shape)

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
            # Concat image and label embeddings
            embeddings = torch.cat((features, init_label_embeddings), 1)

            # zdxy+
            '''
            if self.use_month and self.use_loc:
                shp_t = month_embedding.shape
                month_embedding = month_embedding.view(shp_t[0], 1, shp_t[1])
                loc_embedding = loc_embedding.view(shp_t[0], 1, shp_t[1])
                embeddings = torch.cat((features, month_embedding,
                                        loc_embedding, init_label_embeddings), 1)

            elif self.use_month:
                # print(embeddings.shape,month_embedding.shape)
                shp_t = month_embedding.shape
                month_embedding = month_embedding.view(shp_t[0], 1, shp_t[1])
                embeddings = torch.cat((features, month_embedding, init_label_embeddings), 1)

            elif self.use_loc:
                shp_t = loc_embedding.shape
                loc_embedding = loc_embedding.view(shp_t[0], 1, shp_t[1])
                embeddings = torch.cat((features, loc_embedding, init_label_embeddings), 1)
            '''
        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)
        attns = []
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]
        # output = self.output_linear(label_embeddings)
        # output = self.max_classifier(label_embeddings)
        output = self.group_wise_linear(label_embeddings)
        diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0), 1, 1).cuda()

        # zdxy modified
        # can be another linear
        # output,_ = output.max(dim=2)
        #  $$ output = (output * diag_mask).sum(-1)  ##
        # output = self.mlp(label_embeddings)

        # zdxy+
        # output = self.out_project(torch.relu(output)).view(output.shape[0],output.shape[1])
        # output = self.fc(label_embeddings)

        return output, None, attns, month_out, loc_out



class Multichannel16(nn.Module):
    def __init__(self, args, num_labels, use_lmt, pos_emb=False, layers=3, heads=4, dropout=0.1, int_loss=0,
                 no_x_features=False):
        super(Multichannel16, self).__init__()
        self.use_lmt = use_lmt
        self.use_month = args.use_month

        self.no_x_features = no_x_features  # (for no image features)

        # ResNet backbone
        '''
        if args.backbone == 'res18':
            hidden = 512
            self.backbone1 = Backbone_split_res18(3)
            self.backbone2 = Backbone_split_res18(8)
            self.backbone3 = Backbone_split_res18(5)
        elif args.backbone == 'res34':
            hidden = 512
            self.backbone1 = Backbone_split_res34(3)
            self.backbone2 = Backbone_split_res34(8)
            self.backbone3 = Backbone_split_res34(5)
        elif args.backbone == 'res50':
            hidden = 2048
            self.backbone1 = Backbone_split_res50(3)
            self.backbone2 = Backbone_split_res50(8)
            self.backbone3 = Backbone_split_res50(5)
        elif args.backbone == 'res101':
            hidden = 2048
            self.backbone1 = Backbone_split_res101(3)
            self.backbone2 = Backbone_split_res101(8)
            self.backbone3 = Backbone_split_res101(5)
        elif args.backbone == 'cnn':
            hidden = 512
            self.backbone1 = CNN(3)
            self.backbone2 = CNN(8)
            self.backbone3 = CNN(5)
        '''

        '''
        self.cnn_set = []
        for i in range(16):
            self.cnn_set.append(CNN(1))
        '''
        self.cnn_set = nn.ModuleList([CNN(1) for _ in range(16)])
        # self.backbone_all = Backbone_split_res101(3)

        # hidden = 512  # this should match the backbone output feature size
        # zdxy
        hidden = 512
        self.hidden = hidden
        self.downsample = False
        if self.downsample:
            self.conv_downsample = torch.nn.Conv2d(hidden, hidden, (1, 1))

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
        self.channel_embedding = torch.nn.Embedding(16, hidden, padding_idx=None)
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

        features = []
        for i in range(16):
            feat_2d = self.cnn_set[i].cuda()(
                images[:, i, :, :].view(images.shape[0], 1, images.shape[2], images.shape[3]))  # [N, 512, 8,8]
            feat_1d = feat_2d.view(feat_2d.shape[0], feat_2d.shape[1], feat_2d.shape[2] * feat_2d.shape[3])
            feat_1d = feat_1d.permute(0, 2, 1)

            c_embedding_temp = self.channel_embedding(torch.LongTensor([0]).repeat(feat_1d.shape[0]).cuda())
            c_embedding_temp = c_embedding_temp.view(c_embedding_temp.shape[0], 1, -1)
            c_embedding_temp = c_embedding_temp.repeat(1, feat_1d.shape[1], 1)
            #feat_1d = feat_1d + c_embedding_temp


            # feat_1d = feat_2d.reshape(feat_2d.size(0),1,-1)
            features.append(feat_1d)

        features = torch.cat(features, dim=1)

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
        # features_all = self.backbone_all(images[:,1:4,:,:])

        '''
        if self.downsample:
            features = self.conv_downsample(features)
        '''
        '''
        if self.use_pos_enc:
            # zdxy

            pos_encoding = positionalencoding2d(d_model=self.hidden,
                                                width=features.shape[2],
                                                height=features.shape[3])

            # self.position_encoding(features,torch.zeros(features.size(0),18,18, dtype=torch.bool).cuda())
            # print(features.shape)
            # print(self.position_encoding.shape)
            features = features + self.position_encoding.cuda()


        if self.use_geo_enc:

            geo_encoding = positionalencoding2d(self.hidden,
                                                features.shape[2] * 6,
                                                features.shape[3] * 6)
            pos_encoding = torch.zeros_like(features)
            for idx in range(features.shape[0]):
                loc_x = int((img_loc[0][idx] - 1) * 1.6)
                loc_y = int((img_loc[1][idx] - 1) * 1.6)
                pos_encoding[idx, :, :, :] = geo_encoding[:, loc_y:loc_y + 8, loc_x:loc_x + 8]

            features = features + pos_encoding
        '''
        embeddings = torch.cat((features, init_label_embeddings,), 1)
        # embeddings = features
        pos_embedding = self.position_embedding.weight
        # embeddings += pos_embedding

        # print(features1.shape)

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
            # Concat image and label embeddings
            embeddings = torch.cat((features, init_label_embeddings), 1)

            # zdxy+
            '''
            if self.use_month and self.use_loc:
                shp_t = month_embedding.shape
                month_embedding = month_embedding.view(shp_t[0], 1, shp_t[1])
                loc_embedding = loc_embedding.view(shp_t[0], 1, shp_t[1])
                embeddings = torch.cat((features, month_embedding,
                                        loc_embedding, init_label_embeddings), 1)

            elif self.use_month:
                # print(embeddings.shape,month_embedding.shape)
                shp_t = month_embedding.shape
                month_embedding = month_embedding.view(shp_t[0], 1, shp_t[1])
                embeddings = torch.cat((features, month_embedding, init_label_embeddings), 1)

            elif self.use_loc:
                shp_t = loc_embedding.shape
                loc_embedding = loc_embedding.view(shp_t[0], 1, shp_t[1])
                embeddings = torch.cat((features, loc_embedding, init_label_embeddings), 1)
            '''
        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)
        attns = []
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]
        # output = self.output_linear(label_embeddings)
        # output = self.max_classifier(label_embeddings)
        output = self.group_wise_linear(label_embeddings)
        diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0), 1, 1).cuda()

        # zdxy modified
        # can be another linear
        # output,_ = output.max(dim=2)
        #  $$ output = (output * diag_mask).sum(-1)  ##
        # output = self.mlp(label_embeddings)

        # zdxy+
        # output = self.out_project(torch.relu(output)).view(output.shape[0],output.shape[1])
        # output = self.fc(label_embeddings)

        #if month_out.device == torch.device('cuda:1'):
        #    month_out = torch.zeros(1, 12).cuda()
        #    loc_out = torch.zeros(1, 286).cuda()

        return output, None, attns, month_out, loc_out

class Multichannel_3(nn.Module):
    def __init__(self, args, num_labels, use_lmt, pos_emb=False, layers=3, heads=4, dropout=0.1, int_loss=0,
                 no_x_features=False):
        super(Multichannel_3, self).__init__()
        self.use_lmt = use_lmt
        self.use_month = args.use_month

        self.no_x_features = no_x_features  # (for no image features)

        # ResNet backbone
        '''
        if args.backbone == 'res18':
            hidden = 512
            self.backbone1 = Backbone_split_res18(3)
            self.backbone2 = Backbone_split_res18(8)
            self.backbone3 = Backbone_split_res18(5)
        elif args.backbone == 'res34':
            hidden = 512
            self.backbone1 = Backbone_split_res34(3)
            self.backbone2 = Backbone_split_res34(8)
            self.backbone3 = Backbone_split_res34(5)
        elif args.backbone == 'res50':
            hidden = 2048
            self.backbone1 = Backbone_split_res50(3)
            self.backbone2 = Backbone_split_res50(8)
            self.backbone3 = Backbone_split_res50(5)
        elif args.backbone == 'res101':
            hidden = 2048
            self.backbone1 = Backbone_split_res101(3)
            self.backbone2 = Backbone_split_res101(8)
            self.backbone3 = Backbone_split_res101(5)
        elif args.backbone == 'cnn':
            hidden = 512
            self.backbone1 = CNN(3)
            self.backbone2 = CNN(8)
            self.backbone3 = CNN(5)
        '''

        '''
        self.cnn_set = []
        for i in range(16):
            self.cnn_set.append(CNN(1))
        '''
        self.cnn_set = nn.ModuleList([CNN(1) for _ in range(3)])
        # self.backbone_all = Backbone_split_res101(3)

        # hidden = 512  # this should match the backbone output feature size
        # zdxy
        hidden = 512
        self.hidden = hidden
        self.downsample = False
        if self.downsample:
            self.conv_downsample = torch.nn.Conv2d(hidden, hidden, (1, 1))

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
        self.channel_embedding = torch.nn.Embedding(16, hidden, padding_idx=None)
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

        features = []
        for i in range(3):
            feat_2d = self.cnn_set[i].cuda()(
                images[:, i, :, :].view(images.shape[0], 1, images.shape[2], images.shape[3]))  # [N, 512, 8,8]
            feat_1d = feat_2d.view(feat_2d.shape[0], feat_2d.shape[1], feat_2d.shape[2] * feat_2d.shape[3])
            feat_1d = feat_1d.permute(0, 2, 1)

            c_embedding_temp = self.channel_embedding(torch.LongTensor([0]).repeat(feat_1d.shape[0]).cuda())
            c_embedding_temp = c_embedding_temp.view(c_embedding_temp.shape[0], 1, -1)
            c_embedding_temp = c_embedding_temp.repeat(1, feat_1d.shape[1], 1)
            #feat_1d = feat_1d + c_embedding_temp


            # feat_1d = feat_2d.reshape(feat_2d.size(0),1,-1)
            features.append(feat_1d)

        features = torch.cat(features, dim=1)

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
        # features_all = self.backbone_all(images[:,1:4,:,:])

        '''
        if self.downsample:
            features = self.conv_downsample(features)
        '''
        '''
        if self.use_pos_enc:
            # zdxy

            pos_encoding = positionalencoding2d(d_model=self.hidden,
                                                width=features.shape[2],
                                                height=features.shape[3])

            # self.position_encoding(features,torch.zeros(features.size(0),18,18, dtype=torch.bool).cuda())
            # print(features.shape)
            # print(self.position_encoding.shape)
            features = features + self.position_encoding.cuda()


        if self.use_geo_enc:

            geo_encoding = positionalencoding2d(self.hidden,
                                                features.shape[2] * 6,
                                                features.shape[3] * 6)
            pos_encoding = torch.zeros_like(features)
            for idx in range(features.shape[0]):
                loc_x = int((img_loc[0][idx] - 1) * 1.6)
                loc_y = int((img_loc[1][idx] - 1) * 1.6)
                pos_encoding[idx, :, :, :] = geo_encoding[:, loc_y:loc_y + 8, loc_x:loc_x + 8]

            features = features + pos_encoding
        '''
        embeddings = torch.cat((features, init_label_embeddings,), 1)
        # embeddings = features
        pos_embedding = self.position_embedding.weight
        # embeddings += pos_embedding

        # print(features1.shape)

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
            # Concat image and label embeddings
            embeddings = torch.cat((features, init_label_embeddings), 1)

            # zdxy+
            '''
            if self.use_month and self.use_loc:
                shp_t = month_embedding.shape
                month_embedding = month_embedding.view(shp_t[0], 1, shp_t[1])
                loc_embedding = loc_embedding.view(shp_t[0], 1, shp_t[1])
                embeddings = torch.cat((features, month_embedding,
                                        loc_embedding, init_label_embeddings), 1)

            elif self.use_month:
                # print(embeddings.shape,month_embedding.shape)
                shp_t = month_embedding.shape
                month_embedding = month_embedding.view(shp_t[0], 1, shp_t[1])
                embeddings = torch.cat((features, month_embedding, init_label_embeddings), 1)

            elif self.use_loc:
                shp_t = loc_embedding.shape
                loc_embedding = loc_embedding.view(shp_t[0], 1, shp_t[1])
                embeddings = torch.cat((features, loc_embedding, init_label_embeddings), 1)
            '''
        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)
        attns = []
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]
        # output = self.output_linear(label_embeddings)
        # output = self.max_classifier(label_embeddings)
        output = self.group_wise_linear(label_embeddings)
        diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0), 1, 1).cuda()

        # zdxy modified
        # can be another linear
        # output,_ = output.max(dim=2)
        #  $$ output = (output * diag_mask).sum(-1)  ##
        # output = self.mlp(label_embeddings)

        # zdxy+
        # output = self.out_project(torch.relu(output)).view(output.shape[0],output.shape[1])
        # output = self.fc(label_embeddings)

        #if month_out.device == torch.device('cuda:1'):
        #    month_out = torch.zeros(1, 12).cuda()
        #    loc_out = torch.zeros(1, 286).cuda()

        return output, None, attns, month_out, loc_out






