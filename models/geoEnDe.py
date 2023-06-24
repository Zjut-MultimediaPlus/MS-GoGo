import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math
from pdb import set_trace as stop
from .transformer_layers import SelfAttnLayer,QueryLayer
from .backbone import Backbone,Backbone_split_res101,Backbone_split_res34,Backbone_split_res18,Backbone_split_res50,CNN
from .utils import custom_replace,weights_init
from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from utils.aug_plus import augmentation_plus_gpu
from .position_enc import PositionEmbeddingSine,positionalencoding2d
import torchvision


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
        if len(x.shape)>=3:
            x = x.view(x.shape[0],x.shape[2])
        x = self.net(x)
        x = self.softmax(x)
        return x

class GeoTransEnDe_16c(nn.Module):
    def __init__(self, args, num_labels, layers=3, heads=4, dropout=0.1):
        super(GeoTransEnDe_16c, self).__init__()

        self.no_x_features = args.no_x_features  # (for no image features)
        self.use_geo_emb = args.use_geo_emb
        self.bs = args.batch_size

        # ResNet backbone
        hidden = 0
        self.bb_name = args.backbone
        if args.backbone == 'res18':
            hidden = 512
            self.backbone1 = Backbone_split_res18(3)
            self.backbone2 = Backbone_split_res18(8)
            self.backbone3 = Backbone_split_res18(2)
            self.backbone4 = Backbone_split_res18(3)
        elif args.backbone == 'res34':
            hidden = 512
            self.backbone1 = Backbone_split_res34(3)
            self.backbone2 = Backbone_split_res34(8)
            self.backbone3 = Backbone_split_res34(5)
        elif args.backbone == 'res50':
            hidden = 2048
            self.backbone1 = Backbone_split_res50(3)
            self.backbone2 = Backbone_split_res50(8)
            self.backbone3 = Backbone_split_res50(2)
            self.backbone4 = Backbone_split_res50(3)
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
        elif args.backbone == 'swin':
            self.backbone = SwinTransformer(256,4,16,17,window_size=8)
            hidden = 768
        elif args.backbone == 'swin_v2':
            self.backbone = SwinTransformerV2(256,4,16,17,window_size=8)
            hidden = 768

        self.project1 = nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=(1, 1))
        self.project2 = nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=(1, 1))
        self.project3 = nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=(1, 1))
        self.project4 = nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=(1, 1))
        # self.backbone_all = Backbone_split_res101(3)

        # hidden = 512  # this should match the backbone output feature size
        # zdxy
        self.hidden = hidden
        self.downsample = False
        if self.downsample:
            self.conv_downsample = torch.nn.Conv2d(hidden, hidden, (1, 1))


        # zdxy+
        self.month_embedding = torch.nn.Embedding(12, hidden, padding_idx=None)
        self.month_embedding.apply(weights_init)
        self.loc_embedding = torch.nn.Embedding(11 * 26, hidden, padding_idx=None)
        self.loc_embedding.apply(weights_init)
        self.feature_embedding = torch.nn.Embedding(3, hidden, padding_idx=None)
        self.feature_embedding.apply(weights_init)
        self.channel_embedding = torch.nn.Embedding(4, hidden, padding_idx=None)
        self.channel_embedding.apply(weights_init)




        # Transformer
        '''
        self.self_enc_layers = nn.ModuleList([SelfAttnLayer(hidden, heads, dropout) for _ in range(layers)])
        self.self_dec_layers = nn.ModuleList([QueryLayer(hidden, heads, dropout) for _ in range(layers)])
        '''

        #self.enc_layer = nn.TransformerEncoderLayer(hidden,heads,hidden)
        #self.dec_layer = nn.TransformerDecoderLayer(hidden,heads,hidden)
        #self.trans_encoder = nn.TransformerEncoder(self.enc_layer,layers)
        #self.trans_decoder = nn.TransformerDecoder(self.dec_layer,layers)
        self.trans = nn.Transformer(hidden,heads,args.enc_layers,args.dec_layers,hidden,dropout,norm_first=False,batch_first=True)
        self.encoder_layer = nn.TransformerEncoderLayer(hidden,heads,hidden,dropout,batch_first=True,norm_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer,args.enc_layers,nn.LayerNorm(hidden))
        self.decoder_layer = nn.TransformerDecoderLayer(hidden, heads, hidden, dropout, batch_first=True,norm_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer,args.dec_layers,nn.LayerNorm(hidden))

        # Classifier
        # Output is of size num_labels because we want a separate classifier for each label
        # Output is of size num_labels becase we want a separate classifier for each label
        self.output_linear = torch.nn.Linear(hidden, num_labels)
        self.max_classifier = MaxClassifier(hidden, num_labels)
        self.group_wise_linear = GroupWiseLinear(num_labels, hidden)
        self.enhence_feature_norm = torch.nn.LayerNorm(hidden)
        L = 256
        if 'swin' in self.bb_name:
            L = 64
        else:
            L = 256
        self.enhence_feature_projector = GroupWiseLinear(L,self.hidden,True)
        self.time_decoder = torch.nn.Sequential(GroupWiseLinear(num_labels,hidden),
                                                torch.nn.ReLU(),
                                                torch.nn.Dropout(dropout),
                                                torch.nn.Linear(num_labels,1),
                                                torch.nn.Sigmoid())

        self.loc_decoder = torch.nn.Sequential(GroupWiseLinear(num_labels,hidden),
                                               torch.nn.ReLU(),
                                               torch.nn.Dropout(dropout),
                                               torch.nn.Linear(num_labels,4),
                                               torch.nn.Sigmoid())

        self.label_input = torch.Tensor(np.arange(num_labels)).view(1, -1).long()
        self.label_lt = torch.nn.Embedding(num_labels, hidden, padding_idx=None)

        # zdxy+
        self.out_project = torch.nn.Linear(num_labels, 1)
        self.fc = GroupWiseLinear(num_class=num_labels, hidden_dim=hidden)
        self.mlp = MLP(hidden, num_labels)
        self.coordinate_encoder = torch.nn.Sequential(
            torch.nn.Linear(4,hidden//4),
            #torch.nn.BatchNorm1d(hidden // 4),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear((hidden//4),hidden//2),
            #torch.nn.BatchNorm1d(hidden // 2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
        )
        self.year_time_encoder = torch.nn.Sequential(
            torch.nn.Linear(1,hidden//4),
            #torch.nn.BatchNorm1d(hidden//4),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear((hidden//4),hidden//2),
            #torch.nn.BatchNorm1d(hidden//2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
        )

        self.geo_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden,hidden),
            #torch.nn.BatchNorm1d(hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden,hidden),
            #torch.nn.BatchNorm1d(hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden),
        )

        # Other
        self.LayerNorm_enc = nn.LayerNorm(hidden)
        self.LayerNorm_dec = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        self.label_lt.apply(weights_init)
        #self.known_label_lt.apply(weights_init)
        self.LayerNorm_enc.apply(weights_init)
        self.LayerNorm_dec.apply(weights_init)
        #self.self_enc_layers.apply(weights_init)
        #self.self_dec_layers.apply(weights_init)
        self.output_linear.apply(weights_init)

    def forward(self, images, sl, train=True):

        const_label_input = self.label_input.repeat(images.size(0), 1).cuda()
        init_label_embeddings = self.label_lt(const_label_input)
        assert torch.sum(torch.isnan(init_label_embeddings)) == 0, print(init_label_embeddings)

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

        if 'swin' not in self.bb_name:
            features1 = self.backbone1(group1)
            features2 = self.backbone2(group2)
            features3 = self.backbone3(group3)
            features4 = self.backbone4(group4)

            features1 = features1.view(features1.size(0), features1.size(1), -1).permute(0, 2, 1)
            features2 = features2.view(features2.size(0), features2.size(1), -1).permute(0, 2, 1)
            features3 = features3.view(features3.size(0), features3.size(1), -1).permute(0, 2, 1)
            features4 = features4.view(features3.size(0), features4.size(1), -1).permute(0, 2, 1)
            # features_all = features_all.view(features_all.size(0), features_all.size(1), -1).permute(0, 2, 1)
            # print(features_all.shape)

            img_embedding = torch.cat((features1, features2, features3, features4), 1)

            assert torch.sum(torch.isnan(features1)) == 0, print(features1)
            assert torch.sum(torch.isnan(features2)) == 0, print(features2)
            assert torch.sum(torch.isnan(features3)) == 0, print(features3)
            assert torch.sum(torch.isnan(features4)) == 0, print(features4)
        else:
            features = self.backbone(images)
            #features = features.view(features.size(0), features.size(1), -1).permute(0, 2, 1)
            img_embedding = features


        #features1 = self.project1(features1)
        #features2 = self.project2(features2)
        #features3 = self.project3(features3)
        #features4 = self.project4(features4)

        month_out = None
        loc_out = None

        year_time, y,x,l,w = torch.zeros(self.bs,1).cuda(),torch.zeros(self.bs,1).cuda(),torch.zeros(self.bs,1).cuda(),torch.zeros(self.bs,1),torch.zeros(self.bs,1)

        if self.use_geo_emb:
            year_time = sl[:,0].reshape(-1,1)
            year_time_embedding = self.year_time_encoder(year_time)

            coord = sl[:,1:5].reshape(-1,4)
            coordinate_embedding = self.coordinate_encoder(coord)
            geo_embedding = self.geo_mlp(torch.cat((year_time_embedding,coordinate_embedding),dim=1))

            shp = geo_embedding.shape
            geo_embedding = geo_embedding.reshape([shp[0],1,shp[1]]).repeat(1, init_label_embeddings.shape[1], 1)

            assert torch.sum(torch.isnan(geo_embedding)) == 0, print(geo_embedding)
            label_embeddings = init_label_embeddings + geo_embedding
        else:
            label_embeddings = init_label_embeddings

        # features_all = self.backbone_all(images[:,1:4,:,:])





        # embeddings = features
        #pos_embedding = self.position_embedding.weight
        # embeddings += pos_embedding

        # print(features1.shape)


        # Feed image and label embeddings through Transformer encoder
        '''
        img_embedding = self.LayerNorm_enc(img_embedding)
        enc_attns = []
        for layer in self.self_enc_layers:
            img_embedding, attn = layer(img_embedding, mask=None)
            enc_attns += attn.detach().unsqueeze(0).data

        # Feed image and label embeddings through Transformer encoder
        queries = self.LayerNorm_dec(queries)
        embeddings = img_embedding
        dec_attns = []
        for layer in self.self_dec_layers:
            embeddings, attn = layer(queries,embeddings,embeddings, mask=None)
            dec_attns += attn.detach().unsqueeze(0).data
            
        '''
        #queries = torch.zeros_like(img_embedding)
        #queries[:,0:init_label_embeddings.size(1),] = queries[:,0:init_label_embeddings.size(1),:] + label_embeddings
        #queries = label_embeddings
        queries = label_embeddings
        #embeddings = self.trans(img_embedding,queries)
        #print(img_embedding.shape)
        enhenced_embeddings = self.encoder(img_embedding)
        assert torch.sum(torch.isnan(enhenced_embeddings)) == 0, print(enhenced_embeddings)

        img_rep = self.enhence_feature_projector(self.enhence_feature_norm(enhenced_embeddings))
        assert torch.sum(torch.isnan(img_rep)) == 0, print(img_rep)
        out_seq = self.decoder(enhenced_embeddings,queries)
        assert torch.sum(torch.isnan(out_seq)) == 0, print(out_seq)

        # Readout each label embedding using a linear layer
        #label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]
        label_embeddings = out_seq[:,0:init_label_embeddings.size(1),:]
        output = self.output_linear(label_embeddings)
        # output = self.max_classifier(label_embeddings)
        output = self.group_wise_linear(label_embeddings)
        assert torch.sum(torch.isnan(output)) == 0, print(output)


        loc_original = torch.concat((y, x, l.cuda(), w.cuda()), dim=1)

        time_rebuilt = torch.zeros_like(year_time)
        loc_rebuilt = torch.zeros_like(loc_original)

        if self.use_geo_emb:
            time_rebuilt = self.time_decoder(label_embeddings)
            loc_rebuilt = self.loc_decoder(label_embeddings)

        # zdxy modified
        # can be another linear
        # output,_ = output.max(dim=2)
        # output = (output * diag_mask).sum(-1)  ##
        # output = self.mlp(label_embeddings)

        # zdxy+
        # output = self.out_project(torch.relu(output)).view(output.shape[0],output.shape[1])
        # output = self.fc(label_embeddings)

        return img_rep,output, None, time_rebuilt, loc_rebuilt


    def geo_transform(self,year,month,day,loc_y,loc_x,L,W,step,patch_l,patch_w,range_l,range_w,start_lat,start_lon):
        '''
        transform coordinate into longitude and latitude, time into year time
        the output data are dimansionless
        :param year:
        :param month:
        :param day:
        :param loc_y: coordinate y in image name
        :param loc_x: coordinate x in image name
        :param L: pixel length of whole disc data
        :param W: pixel width of whole disc data
        :param step: pixel step of patch
        :param patch_l: length of patch
        :param patch_w: width of patch
        :param range_l: longitude range of whole disc data
        :param range_w: latitude range of whole disc data
        :param start_lat: the latitude of up-left point of whole disc data
        :param start_lon: the longitude of up-left point of whole disc data
        :return:
        '''

        month_len   = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        month_prior_normal = [0,  31, 59, 90, 120,151,182,213,244,274,305,335]
        month_prior_leap = [0, 31, 59+1, 90+1, 120+1,151+1,182+1,213+1,244+1,274+1,305+1,335+1]

        leap_mark = year%4 == 0

        prior_day = torch.zeros_like(year)
        year_day  = torch.zeros_like(year)
        year_len  = torch.zeros_like(year)
        #print(year_day.shape)
        for i in range(len(leap_mark)):
            if leap_mark[i]:
                prior_day[i] = month_prior_leap[month[i]-1]
                year_len[i] = 366
            else:
                prior_day[i] = month_prior_normal[month[i]-1]
                year_len[i] = 365
            year_day[i] = prior_day[i] + day[i]

        year_time = year_day/year_len


        y = start_lat - ((loc_y*step+0.5*patch_l)/L*range_l)
        x = start_lon + ((loc_x*step+0.5*patch_w)/W*range_w)


        x[x>180] = x[x>180] - 360

        y_result = (y - (-90))/180
        x_result = (x - (-180))/360
        l_result = torch.Tensor([(patch_l/L*range_l)/180]).repeat(year.shape[0],year.shape[1])
        w_result = torch.Tensor([(patch_w/W*range_w)/360]).repeat(year.shape[0],year.shape[1])

        return year_time, y_result, x_result, l_result, w_result


