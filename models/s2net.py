import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import math
from pdb import set_trace as stop
#from .transformer_layers import SelfAttnLayer
#from .backbone import Backbone,Backbone_split_res101,Backbone_split_res34,Backbone_split_res18,Backbone_split_res50,CNN
#from .utils import custom_replace,weights_init
#from .position_enc import PositionEmbeddingSine,positionalencoding2d
#from .swin_transformer import SwinTransformer
#from .swin_transformer_v2 import SwinTransformerV2
from .transformer_layers import SelfAttnLayer
from .utils import custom_replace,weights_init

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
        #print(x.shape, self.W.shape)
        # x: B,K,d
        x = (self.W * x).sum(-1)

        if self.bias:
            x = x + self.b
        return x

class SpectralBranchBlock(nn.Module):
    def __init__(self,in_channels,expend_ratio,down_ratio,groups):
        """

        :param in_channels: the input channel number
        :param expend_ratio: out_channel/in_channel
        :param down_ratio: spatial down sampling rate
        :param groups: the group of conv
        """
        super(SpectralBranchBlock,self).__init__()
        self.spatial_extractor = nn.Conv2d(in_channels=in_channels,
                                           out_channels=in_channels,
                                           kernel_size=(7,7),
                                           padding=(3,3),
                                           groups=in_channels)
        torch.nn.init.xavier_uniform_(self.spatial_extractor.weight)
        self.norm1 = nn.LayerNorm(in_channels)#nn.BatchNorm2d(in_channels)#nn.LayerNorm(in_channels)

        self.spectral_lifter = nn.Conv2d(in_channels=in_channels,
                                            out_channels=in_channels*4,
                                            kernel_size=(1,1),
                                            groups=groups)
        torch.nn.init.xavier_uniform_(self.spectral_lifter.weight)
        self.act = nn.GELU()
        self.spectral_droper = nn.Conv2d(in_channels=in_channels*4,
                                            out_channels=in_channels,
                                            kernel_size=(1,1),
                                            groups=groups)
        torch.nn.init.xavier_uniform_(self.spectral_droper.weight)
        #layer_scale_init_value = 1e-6
        #self.gamma = nn.Parameter(layer_scale_init_value*torch.ones((in_channels)),requires_grad=True) if layer_scale_init_value>0 else None

        self.feature_merger = nn.Conv2d(in_channels=in_channels,
                                        out_channels=in_channels*expend_ratio,
                                        kernel_size=(down_ratio,down_ratio),
                                        stride=(down_ratio,down_ratio),
                                        groups=groups)
        torch.nn.init.xavier_uniform_(self.feature_merger.weight)
        self.norm2 = nn.LayerNorm(in_channels*expend_ratio)#nn.BatchNorm2d(in_channels*expend_ratio)#nn.LayerNorm(in_channels*expend_ratio)

    def forward(self,x):
        # x: (B,C,H,W)
        x = self.spatial_extractor(x).permute(0,2,3,1) # (B,H,W,C)
        #torch.nn.init.xavier_uniform_(self.spatial_extractor.weight)
        #torch.nn.init.xavier_uniform_(self.spatial_extractor.bias, 0)

        x = self.norm1(x).permute(0,3,1,2) # (B,C,H,W)
        x = self.spectral_lifter(x)

        #torch.nn.init.xavier_uniform_(self.spectral_lifter.bias, 0)
        x = self.act(x)
        x = self.spectral_droper(x)

        #torch.nn.init.xavier_uniform_(self.spectral_droper.bias, 0)

        #if self.gamma is not None:
        #    x = self.gamma * x

        x = self.feature_merger(x).permute(0,2,3,1)

        #torch.nn.init.xavier_uniform_(self.feature_merger.bias, 0)
        x = self.norm2(x).permute(0,3,1,2)

        return x

class SpatialBranchBlock(nn.Module):
    def __init__(self,in_channels,expend_ratio,down_ratio):
        super(SpatialBranchBlock,self).__init__()

        self.spatial_extractor = nn.Conv2d(in_channels=in_channels,
                                           out_channels=in_channels,
                                           kernel_size=(7, 7),
                                           padding=(3, 3),
                                           groups=in_channels)
        torch.nn.init.xavier_uniform_(self.spatial_extractor.weight)

        self.norm1 = nn.LayerNorm(in_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                                         out_channels=in_channels * 4,
                                         kernel_size=(1, 1))
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        #torch.nn.init.xavier_uniform_(self.conv1.bias,0)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels=in_channels * 4,
                                         out_channels=in_channels,
                                         kernel_size=(1, 1))
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        #torch.nn.init.xavier_uniform_(self.conv2.bias,0)

        layer_scale_init_value = 1e-6
        #self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_channels)),
        #                          requires_grad=True) if layer_scale_init_value > 0 else None

        self.feature_merger = nn.Conv2d(in_channels=in_channels,
                                        out_channels=in_channels * expend_ratio,
                                        kernel_size=(down_ratio, down_ratio),
                                        stride=(down_ratio, down_ratio),
                                        groups=in_channels)
        torch.nn.init.xavier_uniform_(self.feature_merger.weight)
        #torch.nn.init.xavier_uniform_(self.feature_merger.bias)
        self.norm2 = nn.LayerNorm(in_channels * expend_ratio)

    def forward(self,x):
        #x: B,H,W,C
        # x: (B,C,H,W)
        #print(x.shape)
        x = self.spatial_extractor(x).permute(0, 2, 3, 1)  # (B,H,W,C)
        #print(x.shape)
        x = self.norm1(x).permute(0, 3, 1, 2)  # (B,C,H,W)

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)#.permute(0,2,3,1)
        #if self.gamma is not None:
        #    x = self.gamma * x
        #x = x.permute(0,3,2,1)
        x = self.feature_merger(x).permute(0, 2, 3, 1)
        x = self.norm2(x).permute(0, 3, 1, 2)

        return x

class TwoBranchExtractor(nn.Module):
    def __init__(self,dims=[16,64,128,256,512,1024],groups=16):
        super(TwoBranchExtractor,self).__init__()
        self.stem = nn.Conv2d(dims[0],dims[1],(4,4),(4,4),groups=groups)
        self.norm = nn.LayerNorm(dims[1])

        self.spectral_branch = nn.ModuleList()
        self.spatial_branch = nn.ModuleList()

        for i in range(4):
            self.spectral_branch.append(
                SpectralBranchBlock(dims[i+1],dims[i+2]//dims[i+1],2,groups)
            )
            self.spatial_branch.append(
                SpatialBranchBlock(dims[i+1],dims[i+2]//dims[i+1],2)
            )

    def forward(self,x):
        x = self.stem(x).permute(0,2,3,1)
        spatial_feature = self.norm(x).permute(0,3,1,2)
        spectral_feature = spatial_feature

        for i in range(4):
            spectral_feature = self.spectral_branch[i](spectral_feature)
            spatial_feature = self.spatial_branch[i](spatial_feature)#+spectral_feature

        return spectral_feature,spatial_feature

class SemanticTokenizer(nn.Module):
    def __init__(self, num_tokens, hidden_size, in_channel):
        super(SemanticTokenizer,self).__init__()
        self.L = num_tokens
        self.hidden_size = hidden_size

        self.conv = nn.Conv2d(in_channel, self.L, (1,1), (1,1))

    def forward(self,x):
        b,c,h,w = x.shape
        att_map = self.conv(x)
        att_map = att_map.view([b,self.L,-1]).contiguous()
        att_map = torch.softmax(att_map, dim=-1)
        x = x.view([b,c,-1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', att_map,x)
        return tokens

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

class SSNet(nn.Module):
    def __init__(self, in_channel=16,dims=[16,64,128,256,512,1024], num_sem_tokens=64,hidden_size=1024, heads=4,
                 spectral_encoder_layers=2,spatial_encoder_layers=2,overall_encoder_layers=2,
                 spectral_decoder_layers=1,spatial_decoder_layers=1,num_class=17,
                 use_loc=False,use_month=False):
        super(SSNet,self).__init__()
        self.in_channel = in_channel
        self.num_class = num_class
        self.use_loc = use_loc
        self.use_month = use_month

        self.month_mlp = MLP(hidden_size, 12)
        self.loc_mlp = MLP(hidden_size, 286)

        self.extactor = TwoBranchExtractor(dims,in_channel)
        self.semantic_tokenizer = SemanticTokenizer(num_sem_tokens,hidden_size,dims[-1])
        self.spectral_projector = nn.Conv2d(dims[-1],dims[-1]*in_channel,(4,4),(4,4))

        self.label_embeddings = torch.nn.Embedding(num_class, hidden_size, padding_idx=None)
        self.month_embedding = torch.nn.Embedding(12, hidden_size, padding_idx=None)
        self.loc_embedding = torch.nn.Embedding(11 * 26, hidden_size, padding_idx=None)

        self.enc_norm1 = nn.LayerNorm(hidden_size)
        self.enc_norm2 = nn.LayerNorm(hidden_size)
        self.enc_norm3 = nn.LayerNorm(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size,heads,batch_first=True,norm_first=False)
        self.spectral_mixer = nn.TransformerEncoder(encoder_layer,spectral_encoder_layers)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, heads, batch_first=True, norm_first=False)
        self.spatial_mixer = nn.TransformerEncoder(encoder_layer, spatial_encoder_layers)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, heads, batch_first=True, norm_first=False)
        #self.overall_mixer = nn.TransformerEncoder(encoder_layer, overall_encoder_layers)
        # Transformer
        self.overall_mixer = nn.ModuleList([SelfAttnLayer(hidden_size, heads, 0.1) for _ in range(overall_encoder_layers)])

        self.pos_emb1 = nn.Embedding(16, hidden_size)
        self.pos_emb2 = nn.Embedding(64, hidden_size)

        self.dec_norm1 = nn.LayerNorm(hidden_size)
        self.dec_norm2 = nn.LayerNorm(hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(hidden_size, heads, batch_first=True)
        self.decoder1 = nn.TransformerDecoder(decoder_layer,spectral_decoder_layers,nn.LayerNorm(hidden_size))
        decoder_layer = nn.TransformerDecoderLayer(hidden_size, heads, batch_first=True)
        self.decoder2 = nn.TransformerDecoder(decoder_layer,spatial_decoder_layers,nn.LayerNorm(hidden_size))
        self.gwl1 = GroupWiseLinear(num_class,hidden_size,bias=True)
        self.gwl2 = GroupWiseLinear(num_class,hidden_size,bias=True)

        #self.spatial_trans = nn.Transformer(hidden_size,heads,custom_enc)

        self.semantic_pool = nn.AdaptiveAvgPool1d(1)
        self.spectral_pool = nn.AdaptiveAvgPool1d(1)

        self.semantic_linear= nn.Linear(in_features=hidden_size, out_features=17)
        self.spectral_linear= nn.Linear(in_features=hidden_size, out_features=17)
        self.overall_linear = nn.Linear(in_features=hidden_size, out_features=17)
        #self.system_projector = GroupWiseLinear(8, hidden_size)
        #self.cloud_projector = GroupWiseLinear(9, hidden_size)
        self.system_rep_predictor = Predictor(hidden_size, hidden_size//4)
        self.cloud_rep_predictor = Predictor(hidden_size, hidden_size//4)

        # init
        self.label_embeddings.apply(weights_init)
        self.month_embedding.apply(weights_init)
        self.loc_embedding.apply(weights_init)
        self.enc_norm3.apply(weights_init)
        self.overall_mixer.apply(weights_init)
        self.gwl1.apply(weights_init)
        self.gwl2.apply(weights_init)


    def forward(self,x,month,loc_num):
        spectral_feature,spatial_feature = self.extactor(x)
        semantic_tokens = self.semantic_tokenizer(spatial_feature)
        spectral_tokens = self.spectral_projector(spectral_feature)
        b,c,_,_ = spectral_tokens.shape
        spectral_tokens = spectral_tokens.reshape(b,x.shape[1],c//x.shape[1])

        semantic_embeddings = semantic_tokens + self.pos_emb2.weight.repeat(x.size(0),1,1)
        #self.spatial_mixer(self.enc_norm1(semantic_tokens + self.pos_emb2.weight.repeat(x.size(0),1,1)))
        spectral_embeddings = spectral_tokens + self.pos_emb1.weight.repeat(x.size(0),1,1)
        #self.spectral_mixer(self.enc_norm2()
        #print(self.label_embeddings.weight.repeat(x.size(0), 1).shape)

        queries = self.label_embeddings.weight.repeat(x.size(0), 1,1)

        month_out = None
        loc_out = None

        if self.use_month:
            month_embedding = self.month_embedding(month.long().cuda())
            # print(month)
            # month_out = self.month_mlp(month_embedding)
            # month_out = self.month_mlp(self.month_embedding.weight.view(12, 1, -1))  # for month_loss computation
            # print('month:', self.month_embedding.weight.shape)
            month_embedding = month_embedding.view(month_embedding.shape[0], 1, -1)
            month_embedding = month_embedding.repeat(1, queries.shape[1], 1)
            queries += month_embedding

            month_out = self.month_mlp(self.month_embedding.weight.view(12, 1, -1))


        if self.use_loc:
            loc_embedding = self.loc_embedding(loc_num.long().cuda())
            # loc_out = self.loc_mlp(loc_embedding)
            # print('loc:', self.loc_embedding.weight.shape)
            # loc_out = self.loc_mlp(self.loc_embedding.weight.view(286, 1, -1))  # for loc_loss computation
            loc_embedding = loc_embedding.view(loc_embedding.shape[0], 1, -1)
            loc_embedding = loc_embedding.repeat(1, queries.shape[1], 1)
            queries+= loc_embedding

            loc_out = self.loc_mlp(self.loc_embedding.weight.view(286, 1, -1))

        embeddings = torch.cat((spectral_embeddings, semantic_embeddings, queries),dim=-2)

        attns = []
        for layer in self.overall_mixer:
            embeddings, attn = layer(embeddings, mask=None)
            attns += attn.detach().unsqueeze(0).data
        out_seq = embeddings #self.overall_mixer(self.enc_norm3(t_embeddings))


        semantic_results = self.decoder1(self.dec_norm1(queries),semantic_embeddings)
        #print(queries.shape,semantic_embeddings.shape)
        spectral_results = self.decoder2(self.dec_norm2(queries),spectral_embeddings)

        semantic_rep = semantic_results.mean(dim=-2)
        #semantic_rep = semantic_embeddings.mean(dim=-2)
        spectral_rep = spectral_results.mean(dim=-2)
        #spectral_rep = spectral_embeddings.mean(dim=-2)#self.spectral_pool(spectral_embeddings)

        #system_cat = self.semantic_linear(semantic_rep)#self.system_projector(semantic_embeddings)#self.semantic_linear(semantic_rep)
        #system_cat  = self.gwl1(semantic_results)
        #cloud_cat = self.gwl2(spectral_results)
        #cloud_cat  = self.spectral_linear(spectral_rep)#self.cloud_projector(spectral_embeddings)#self.spectral_linear(spectral_rep)

        system_z = semantic_rep
        system_p = self.system_rep_predictor(system_z)
        cloud_z = spectral_rep
        cloud_p = self.system_rep_predictor(cloud_z)

        results = self.gwl1(out_seq[:,-self.num_class:,:])
        #self.overall_linear(out_seq[:,-self.num_class:,:].mean(dim=-2))#self.gwl1(out_seq[:,-self.num_class:,:])

        #results  = torch.max(system_cat,cloud_cat)
        #results = torch.cat((system_cat,cloud_cat),dim=-1)

        return (system_z.detach(),system_p,cloud_z.detach(),cloud_p), results, month_out,loc_out


if __name__ == '__main__':
    arr = torch.randn(64,16,256,256)
    net = SSNet()

    out = net(arr)
    print(out.shape)



