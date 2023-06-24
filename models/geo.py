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
        print(x.shape)
        x = self.net(x)
        x = self.softmax(x)
        return x


class GeoTrans_16c(nn.Module):
    def __init__(self, args, num_labels, layers=3, heads=4, dropout=0.1):
        super(GeoTrans_16c, self).__init__()

        self.no_x_features = args.no_x_features  # (for no image features)
        self.use_geo_emb = args.use_geo_emb
        self.bs = args.batch_size
        # ResNet backbone
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
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(hidden, heads, dropout) for _ in range(layers)])

        # Classifier
        # Output is of size num_labels because we want a separate classifier for each label
        # Output is of size num_labels becase we want a separate classifier for each label
        self.output_linear = torch.nn.Linear(hidden, num_labels)
        self.max_classifier = MaxClassifier(hidden, num_labels)
        self.group_wise_linear = GroupWiseLinear(num_labels, hidden)
        self.time_decoder = torch.nn.Sequential(GroupWiseLinear(num_labels,hidden),
                                                torch.nn.ReLU(),
                                                torch.nn.Dropout(dropout),
                                                torch.nn.Linear(num_labels,1))

        self.loc_decoder = torch.nn.Sequential(GroupWiseLinear(num_labels,hidden),
                                               torch.nn.ReLU(),
                                               torch.nn.Dropout(dropout),
                                               torch.nn.Linear(num_labels,4))

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
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        self.label_lt.apply(weights_init)
        #self.known_label_lt.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)

    def forward(self, images, year, month, day, loc_y,loc_x):
        const_label_input = self.label_input.repeat(images.size(0), 1).cuda()
        init_label_embeddings = self.label_lt(const_label_input)

        channel1 = images[:, 0, :, :].view(images.shape[0], 1, images.shape[2], images.shape[3])
        channel2 = images[:, 1, :, :].view(images.shape[0], 1, images.shape[2], images.shape[3])
        channel3 = images[:, 2, :, :].view(images.shape[0], 1, images.shape[2], images.shape[3])

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

        month_out = None
        loc_out = None

        year_time, y,x,l,w = torch.zeros(self.bs,1).cuda(),torch.zeros(self.bs,1).cuda(),torch.zeros(self.bs,1).cuda(),torch.zeros(self.bs,1),torch.zeros(self.bs,1)

        if self.use_geo_emb:
            year_time, y,x,l,w = self.geo_transform(year,month,day,loc_y,loc_x,6000,6000,200,1000,1000,120,120,60,80)
            year_time_embedding = self.year_time_encoder(year_time)

            coordinate_embedding = self.coordinate_encoder(torch.cat((y,x,l.cuda(),w.cuda()),dim=1))
            geo_embedding = self.geo_mlp(torch.cat((year_time_embedding,coordinate_embedding),dim=1))

            shp = geo_embedding.shape
            geo_embedding = geo_embedding.reshape([shp[0],1,shp[1]]).repeat(1, init_label_embeddings.shape[1], 1)

            label_embeddings = geo_embedding+init_label_embeddings
        else:
            label_embeddings = init_label_embeddings

        # features_all = self.backbone_all(images[:,1:4,:,:])



        features1 = features1.view(features1.size(0), features1.size(1), -1).permute(0, 2, 1)
        features2 = features2.view(features2.size(0), features2.size(1), -1).permute(0, 2, 1)
        features3 = features3.view(features3.size(0), features3.size(1), -1).permute(0, 2, 1)
        features4 = features4.view(features3.size(0), features4.size(1), -1).permute(0, 2, 1)
        # features_all = features_all.view(features_all.size(0), features_all.size(1), -1).permute(0, 2, 1)
        # print(features_all.shape)

        group1_embedding = self.channel_embedding(torch.LongTensor([0]).repeat(features1.shape[0]).cuda())
        group1_embedding = group1_embedding.view(group1_embedding.shape[0], 1, -1)
        group1_embedding = group1_embedding.repeat(1, features1.shape[1], 1)
        # features1 = features1 + group1_embedding

        group2_embedding = self.channel_embedding(torch.LongTensor([1]).repeat(features1.shape[0]).cuda())
        group2_embedding = group2_embedding.view(group2_embedding.shape[0], 1, -1)
        group2_embedding = group2_embedding.repeat(1, features2.shape[1], 1)
        # features2 = features2 + group2_embedding

        group3_embedding = self.channel_embedding(torch.LongTensor([2]).repeat(features1.shape[0]).cuda())
        group3_embedding = group3_embedding.view(group3_embedding.shape[0], 1, -1)
        group3_embedding = group3_embedding.repeat(1, features3.shape[1], 1)
        # features3 = features3 + group3_embedding

        group4_embedding = self.channel_embedding(torch.LongTensor([3]).repeat(features1.shape[0]).cuda())
        group4_embedding = group4_embedding.view(group3_embedding.shape[0], 1, -1)
        group4_embedding = group4_embedding.repeat(1, features3.shape[1], 1)
        # features4 = features4 + group4_embedding

        features = torch.cat((features1, features2, features3, features4), 1)
        embeddings = torch.cat((features, label_embeddings), 1)
        # embeddings = features
        #pos_embedding = self.position_embedding.weight
        # embeddings += pos_embedding

        # print(features1.shape)

        if self.no_x_features:
            embeddings = init_label_embeddings
        else:
            # Concat image and label embeddings
            embeddings = torch.cat((features, label_embeddings), 1)


        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)
        attns = []
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]
        output = self.output_linear(label_embeddings)
        # output = self.max_classifier(label_embeddings)
        output = self.group_wise_linear(label_embeddings)


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

        return output, None, attns,year_time, loc_original,time_rebuilt, loc_rebuilt


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

