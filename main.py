import numpy.random
import torch
import torch.nn as nn
import argparse,math,numpy as np

import torchvision.models

import models.backbone
from load_data import get_data
# from models import CTranModel,classical
# from models.CTran import CTranModel, CTranModel_16c, CTranModel_split, CTranModel_split_16c,Multichannel3,Multichannel16,Together,Together_3,Multichannel_3
from models.ms_gogo import MS_GoGo
# from models.s2net import SSNet
# from models import CTranModelCub,add_gcn,cnn_rnn
# from models.CTran_original import CTranModel_
# from models.geo import GeoTrans_16c
# from models.geoEnDe import GeoTransEnDe_16c
# from models .geoEn import GeoTransEn_16c
from config_args import get_args
import utils.evaluate as evaluate
import utils.logger as logger
from pdb import set_trace as stop
from optim_schedule import WarmupLinearSchedule
from run_epoch import run_epoch
import os
# from models.Q2L_lib.models.query2label import build_q2l
from utils.vis import vis_2d_tensor,label_tsne_vis,vis_matrix


torch.autograd.set_detect_anomaly(True)

args = get_args(argparse.ArgumentParser())

print('Labels: {}'.format(args.num_labels))
print('Train Known: {}'.format(args.train_known_labels))
print('Test Known:  {}'.format(args.test_known_labels))

train_loader,valid_loader,test_loader = get_data(args)

if args.model == 'ctran':# togethers
    if args.dataset == 'cub':
        #model = CTranModelCub(args,args.num_labels,args.use_lmt,args.pos_emb,args.layers,args.heads,args.dropout,args.no_x_features)
        #print(model.self_attn_layers)
        model = Together_3(args, args.num_labels, args.use_lmt, args.pos_emb, args.layers, args.heads, args.dropout,
                         args.no_x_features)
    else:
        model = Together_3(args,args.num_labels,args.use_lmt,args.pos_emb,args.layers,args.heads,args.dropout,args.no_x_features)
        print(model.self_attn_layers)
elif args.model=='split':# MC_3
    #model = CTranModel_split(args, args.num_labels, args.use_lmt, args.pos_emb, args.layers, args.heads, args.dropout,args.no_x_features)
    model = Multichannel_3(args, args.num_labels, args.use_lmt, args.pos_emb, args.layers, args.heads, args.dropout,
                           args.no_x_features)
elif args.model=='ctran_16c':
    model = CTranModel_16c(args, args.num_labels, args.use_lmt, args.pos_emb, args.layers, args.heads, args.dropout,
                             args.no_x_features)
elif args.model=='ms_gogo':
    model = MS_GoGo(args, args.num_labels, args.use_lmt, args.pos_emb, args.layers, args.heads, args.dropout,
                    args.no_x_features)
elif args.model == 'together':
    model = Together(args, args.num_labels, args.use_lmt, args.pos_emb, args.layers, args.heads, args.dropout,
                             args.no_x_features)
elif args.model == 'mc16':
    model = Multichannel16(args, args.num_labels, args.use_lmt, args.pos_emb, args.layers, args.heads, args.dropout,
                          args.no_x_features)
elif args.model == 'original':
    model = CTranModel_(args,17,args.use_lmt,layers=args.layers,heads=args.heads)
elif args.model == 'q2l':
    model = build_q2l(args)
elif args.model == 'geo':
    model = GeoTrans_16c(args,args.num_labels,2,4,0.1)
elif args.model == 'geo_ende':
    model = GeoTransEnDe_16c(args, args.num_labels, 2, 4, 0.1)
elif args.model == 'geo_en':
    model = GeoTransEn_16c(args, args.num_labels, 2, 4, 0.1)
elif args.model == 'ssnet':
    model = SSNet(spectral_encoder_layers=args.enc_layers,spatial_encoder_layers=args.enc_layers,
                  spectral_decoder_layers=args.dec_layers,spatial_decoder_layers=args.dec_layers,
                  use_month=args.use_month,use_loc=args.use_loc)

    '''
elif args.model == 'mc16':
    model = Multichannel3(args, args.num_labels, args.use_lmt, args.pos_emb, args.layers, args.heads, args.dropout,
                             args.no_x_features)
'''
elif args.model == 'cnn_rnn':
    model = cnn_rnn.CNN_RNN(512,512,17,args.layers)

elif args.model == 'res18':
    model = classical.RES_18(args.num_labels,False)
elif args.model == 'res50':
    model = classical.RES_50(args.num_labels,False)
elif args.model == 'res101':
    model = classical.RES_101(args.num_labels,False)
elif args.model == 'res152':
    model = classical.RES_101(args.num_labels,False)
elif args.model == 'vgg16':
    model = classical.VGG16(args.num_labels, False)
elif args.model == 'vgg19':
    model = classical.VGG19(args.num_labels,False)
elif args.model == 'alex':
    model = classical.ALEX(args.num_labels,False)
elif args.model == 'effb7':
    model = classical.EFFB7(args.num_labels, False)
elif args.model == 'add_gcn':
    if args.backbone == 'res18':
        model = add_gcn.ADD_GCN('res18',torchvision.models.resnet18(), args.num_labels, 512)
    elif args.backbone == 'res101':
        model = add_gcn.ADD_GCN('res101',torchvision.models.resnet101(), args.num_labels, 2048)
    elif args.backbone == 'cnn':
        model = add_gcn.ADD_GCN('cnn', models.backbone.CNN(16), args.num_labels, 512)


def load_saved_model(saved_model_name,model):
    print(saved_model_name,123)
    model_path = './results/'+ saved_model_name+'/best_model.pt'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

print(args.model_name)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
if torch.cuda.device_count() > 1:
    # print("Using", torch.cuda.device_count(), "GPUs!")
    #model = nn.DataParallel(model)
    device = torch.device('cuda:%d' % args.device)
    torch.cuda.set_device(device)
    #device = None
model = model.cuda()

if args.inference:
    model = load_saved_model(args.saved_model_name,model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr,weight_decay=0.0004)#, weight_decay=0.0004)

    # evaluate
    if test_loader is not None:
        print('test_is_not_Known')
        data_loader =test_loader
    else:
        data_loader =valid_loader

    all_preds,all_targs,all_masks,all_ids,test_loss,test_loss_unk,attens = run_epoch(args,model,data_loader,optimizer,False, 1,'Testing')
    test_metrics = evaluate.compute_metrics(args,all_preds,all_targs,all_masks,test_loss,test_loss_unk,0,args.test_known_labels)
    evaluate.print_metrics(test_metrics)


    exit(0)

if args.freeze_backbone:
    for p in model.module.backbone.parameters():
        p.requires_grad=False
    for p in model.module.backbone.base_network.layer4.parameters():
        p.requires_grad=True

if args.optim == 'adam':
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr,weight_decay=0.0004)#, weight_decay=0.0004)
elif args.optim == 'sgd':
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)
elif args.optim == 'ada':
    optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

if args.warmup_scheduler:
    step_scheduler = None
    scheduler_warmup = WarmupLinearSchedule(optimizer, 10, 100)
else:
    scheduler_warmup = None
    patience = 3
    if args.semi_supervise:
        patience=1
        print(patience,'patience')
    if args.scheduler_type == 'plateau' and args.plateau_on=='loss':
        step_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=args.reduce_factor,patience=patience)
    elif args.scheduler_type == 'plateau' and (args.plateau_on == 'map' or args.plateau_on == 'sub_acc'):
        step_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.reduce_factor,
                                                                    patience=patience)
    elif args.scheduler_type == 'step':
        step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.reduce_factor)
    else:
        step_scheduler = None

metrics_logger = logger.Logger(args)
loss_logger = logger.LossLogger(args.model_name)

for epoch in range(1,args.epochs+1):
    print('======================== {} ========================'.format(epoch))
    for param_group in optimizer.param_groups:
        print('LR: {}'.format(param_group['lr']))

    train_loader.dataset.epoch = epoch

    # log the original accuracy

    # if epoch == 1:
    #     ################### Valid #################
    #     all_preds, all_targs, all_masks, all_ids, valid_loss, valid_loss_unk,attens= run_epoch(args, model, valid_loader, None,
    #                                                                                      None, epoch-1, 'Validating',
    #                                                                                      warmup_scheduler=scheduler_warmup,
    #                                                                                      device=device)
    #     valid_metrics = evaluate.compute_metrics(args, all_preds, all_targs, all_masks, valid_loss, valid_loss_unk, 0,
    #                                              args.test_known_labels)
    #     loss_logger.log_losses('valid.log', epoch-1, valid_loss, valid_metrics, valid_loss_unk)
    #
    #     ################### Test #################
    #     if test_loader is not None:
    #         all_preds, all_targs, all_masks, all_ids, test_loss, test_loss_unk, attens = run_epoch(args, model, test_loader, None,
    #                                                                                        None, epoch-1, 'Testing',
    #                                                                                        warmup_scheduler=scheduler_warmup,
    #                                                                                        device=device)
    #         test_metrics = evaluate.compute_metrics(args, all_preds, all_targs, all_masks, test_loss, test_loss_unk, 0,
    #                                                 args.test_known_labels)
    #     else:
    #         test_loss, test_loss_unk, test_metrics = valid_loss, valid_loss_unk, valid_metrics
    #     loss_logger.log_losses('test.log', epoch-1, test_loss, test_metrics, test_loss_unk)

    ################### Train #################

    all_preds,all_targs,all_masks,all_ids,train_loss,train_loss_unk,attens = run_epoch(args,model,train_loader,None,optimizer,epoch,'Training',train=True,warmup_scheduler=scheduler_warmup,device=device)

    train_metrics = evaluate.compute_metrics(args, all_preds, all_targs, all_masks, train_loss, train_loss_unk, 0,
                                             args.train_known_labels)
    loss_logger.log_losses('train.log', epoch, train_loss, train_metrics, train_loss_unk)



    ################### Valid #################
    all_preds,all_targs,all_masks,all_ids,valid_loss,valid_loss_unk,attens = run_epoch(args,model,valid_loader,None,None,epoch,'Validating',warmup_scheduler=scheduler_warmup,device=device)
    valid_metrics = evaluate.compute_metrics(args,all_preds,all_targs,all_masks,valid_loss,valid_loss_unk,0,args.test_known_labels)
    loss_logger.log_losses('valid.log',epoch,valid_loss,valid_metrics,valid_loss_unk)

    ################### Test #################
    if test_loader is not None:
        all_preds,all_targs,all_masks,all_ids,test_loss,test_loss_unk,attens = run_epoch(args,model,test_loader,None,None,epoch,'Testing',warmup_scheduler=scheduler_warmup,device=device)
        test_metrics = evaluate.compute_metrics(args,all_preds,all_targs,all_masks,test_loss,test_loss_unk,0,args.test_known_labels)
    else:
        test_loss,test_loss_unk,test_metrics = valid_loss,valid_loss_unk,valid_metrics

    loss_logger.log_losses('test.log',epoch,test_loss,test_metrics,test_loss_unk)
    if args.warmup_scheduler:
        scheduler_warmup.step(epoch=epoch)
    elif step_scheduler is not None:
        if args.scheduler_type == 'step':
            step_scheduler.step(epoch)
        elif args.scheduler_type == 'plateau':
            if args.plateau_on == 'loss':
                step_scheduler.step(valid_loss_unk)
            elif args.plateau_on == 'map':
                step_scheduler.step(valid_metrics['mAP'])
            elif args.plateau_on == 'sub_acc':
                step_scheduler.step(valid_metrics['ACC'])


    ############## Log and Save ##############
    #def evaluate(self, train_metrics, valid_metrics, test_metrics, epoch, model, valid_loss, test_loss, args):
    best_valid,best_test = metrics_logger.evaluate(train_metrics,valid_metrics,test_metrics,epoch,model,attens,valid_loss,test_loss,args,'test.log')

    print(args.model_name)
