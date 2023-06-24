import argparse,math,time,warnings,copy, numpy as np, os.path as path


import torch, torch.nn as nn, torch.nn.functional as F
from pdb import set_trace as stop
from tqdm import tqdm
from models.utils import custom_replace
import random
from itertools import cycle

class DataPrefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if k != 'meta' and isinstance(self.batch[k],torch.Tensor):
                    self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

def cosine_dis(vector_1,vector_2,batch_size):
    up = (vector_1 * vector_2)
    down = (((vector_1 ** 2).sum(dim=1) ** (0.5)) * ((vector_2 ** 2).sum(dim=1) ** 0.5)).view(-1, 1)
    result = (up / (down + 1e-8)).sum() / batch_size
    return result







def run_epoch(args,model,data,udata,optimizer,epoch,desc,train=False,warmup_scheduler=None,device=None):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    # pre-allocate full prediction and target tensors
    all_predictions = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_targets = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_masks = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_image_ids = []

    max_samples = args.max_samples

    batch_idx = 0
    loss_total = 0
    unk_loss_total = 0

    if args.semi_supervise and train:
        pbar = tqdm(udata, mininterval=0.5, desc=desc, leave=False, ncols=200)
    else:
        pbar = tqdm(data,mininterval=0.5,desc=desc,leave=False,ncols=200)


    #month_pred = None
    if args.semi_supervise and train:
        ldata_iterator = iter(data)


    #print('for batch in bar...')
    for batch in pbar:

        if batch_idx == max_samples:
            break

        if args.semi_supervise and train:
            ubatch = batch

            #print('read batch')
            try:
                lbatch = next(ldata_iterator)
            except StopIteration:
                ldata_iterator = iter(data)
                lbatch = next(ldata_iterator)
                print('next iteration')

            #lbatch, ubatch = batch[0],batch[1]

            #print(ubatch,'hhh')
            if 'geo' in args.model:
                ubatch_1 = ubatch[0]
                ubatch_2 = ubatch[1]
                #ubatch_2 = ubatch[1]
                #print(type(lbatch), type(ubatch_1), type(ubatch_2))

                # labeled data
                limages = lbatch['image']
                labels = lbatch['labels']
                l_sl = lbatch['time_and_location']

                # unlabeled data
                uimages_1 = ubatch_1['image']
                uimages_2 = ubatch_2['image']
                #print((uimages_1-uimages_2).sum())
                u_sl = ubatch_1['time_and_loc']  # same time(day-level) and location for two version
            else :
                ubatch_1 = ubatch[0]
                ubatch_2 = ubatch[1]
                # ubatch_2 = ubatch[1]
                # print(type(lbatch), type(ubatch_1), type(ubatch_2))

                # labeled data
                limages = lbatch['image']
                labels = lbatch['labels']
                month = lbatch['month']
                loc_num = lbatch['loc_num']

                # unlabeled data
                uimages_1 = ubatch_1['image']
                uimages_2 = ubatch_2['image']
                # print((uimages_1-uimages_2).sum())
                #u_sl = ubatch_1['time_and_loc']  # same time(day-level) and location for two version
                umonth = ubatch_1['month']
                uloc_num = ubatch_1['loc_num']

        else:
            # labeled data
            if 'geo' in args.model:
                limages = batch['image']
                labels = batch['labels']
                l_sl = batch['time_and_location']

            else:
                # labeled data
                limages = batch['image']
                labels = batch['labels']
                month = batch['month']
                loc_num = batch['loc_num']

                month_all = torch.from_numpy(np.array(range(12))).view(12, 1)
                loc_all = torch.from_numpy(np.array(range(286))).view(286, 1)
                l_month_label_all = torch.zeros(12, 12).scatter_(1, month_all, 1)
                l_loc_label_all = torch.zeros(286, 286).scatter_(1, loc_all, 1)




        #print(month_num.device)
        


        if args.model=='geo_ende' or args.model=='geo_en':

            if train:
                #print('go through the model')
                l_img_rep, l_pred, l_int_pred, l_time_rebuilt, l_loc_rebuilt = model(limages.cuda(),l_sl.cuda(),train)
                assert torch.sum(torch.isnan(l_img_rep)) == 0 , print(l_img_rep)
                assert torch.sum(torch.isnan(l_pred)) == 0, print(l_pred)
                #assert torch.sum(torch.isnan(l_int_pred)) == 0
                assert torch.sum(torch.isnan(l_time_rebuilt)) == 0, print(l_time_rebuilt)
                assert torch.sum(torch.isnan(l_loc_rebuilt)) == 0, print(l_loc_rebuilt)

                if args.semi_supervise:
                    u_img_rep_1, u_pred_1, u_int_pred_1, u_time_rebuilt_1, u_loc_rebuilt_1 = model(uimages_1.cuda(), u_sl.cuda(),train)
                    u_img_rep_2, u_pred_2, u_int_pred_2, u_time_rebuilt_2, u_loc_rebuilt_2 = model(uimages_2.cuda(), u_sl.cuda(),train)

                    assert torch.sum(torch.isnan(u_img_rep_1)) == 0, print(u_img_rep_1)
                    assert torch.sum(torch.isnan(u_pred_1)) == 0, print(u_pred_1)
                    #assert torch.sum(torch.isnan(u_int_pred_1)) == 0
                    assert torch.sum(torch.isnan(u_time_rebuilt_1)) == 0,print(u_time_rebuilt_1)
                    assert torch.sum(torch.isnan(u_loc_rebuilt_1)) == 0,print(u_loc_rebuilt_1)

                    assert torch.sum(torch.isnan(u_img_rep_2)) == 0,print(u_img_rep_2)
                    assert torch.sum(torch.isnan(u_pred_2)) == 0, print(u_pred_2)
                    #assert torch.sum(torch.isnan(u_int_pred_2)) == 0
                    assert torch.sum(torch.isnan(u_time_rebuilt_2)) == 0,print(u_time_rebuilt_2)
                    assert torch.sum(torch.isnan(u_loc_rebuilt_2)) == 0,print(u_loc_rebuilt_2)

            else:
                with torch.no_grad():
                    l_img_rep, l_pred, l_int_pred, l_time_rebuilt, l_loc_rebuilt = model(limages.cuda(), l_sl.cuda(),not train)
                '''
                    if args.semi_supervise:
                        u_img_rep_1, u_pred_1, u_int_pred_1, u_time_rebuilt_1, u_loc_rebuilt_1 = model(
                                uimages_1.cuda(), u_sl.cuda())
                        u_img_rep_2, u_pred_2, u_int_pred_2, u_time_rebuilt_2, u_loc_rebuilt_2 = model(
                                uimages_2.cuda(), u_sl.cuda())
                '''

        elif args.model == 'ms_gogo':
            if train:
                #print('go through the model')
                l_pred, _, attns, l_time_rebuilt, l_loc_rebuilt = model(limages.cuda(), None,None,month,loc_num)
                #print(type(l_img_rep),type(l_pred),type(l_int_pred),type(attns),type(l_time_rebuilt),type(l_loc_rebuilt))
                #assert torch.sum(torch.isnan(l_img_rep)) == 0 , print(l_img_rep)
                #assert torch.sum(torch.isnan(l_pred)) == 0, print(l_pred)
                #assert torch.sum(torch.isnan(l_int_pred)) == 0
                #assert torch.sum(torch.isnan(l_time_rebuilt)) == 0, print(l_time_rebuilt)
                #assert torch.sum(torch.isnan(l_loc_rebuilt)) == 0, print(l_loc_rebuilt)

                if args.semi_supervise:
                    u_img_rep_1, u_pred_1, u_int_pred_1, attns, u_time_rebuilt_1, u_loc_rebuilt_1 = model(uimages_1.cuda(), None, None, umonth, uloc_num)
                    u_img_rep_2, u_pred_2, u_int_pred_2, attns, u_time_rebuilt_2, u_loc_rebuilt_2 = model(uimages_2.cuda(), None, None, umonth, uloc_num)

                    img_z_1, img_p_1, sem_z_1, sem_p_1 = u_img_rep_1
                    img_z_2, img_p_2, sem_z_2, sem_p_2 = u_img_rep_2

                    #assert torch.sum(torch.isnan(u_img_rep_1)) == 0, print(u_img_rep_1)
                    #assert torch.sum(torch.isnan(u_pred_1)) == 0, print(u_pred_1)
                    #assert torch.sum(torch.isnan(u_int_pred_1)) == 0
                    #assert torch.sum(torch.isnan(u_time_rebuilt_1)) == 0,print(u_time_rebuilt_1)
                    #assert torch.sum(torch.isnan(u_loc_rebuilt_1)) == 0,print(u_loc_rebuilt_1)

                    #assert torch.sum(torch.isnan(u_img_rep_2)) == 0,print(u_img_rep_2)
                    #assert torch.sum(torch.isnan(u_pred_2)) == 0, print(u_pred_2)
                    #assert torch.sum(torch.isnan(u_int_pred_2)) == 0
                    #assert torch.sum(torch.isnan(u_time_rebuilt_2)) == 0,print(u_time_rebuilt_2)
                    #assert torch.sum(torch.isnan(u_loc_rebuilt_2)) == 0,print(u_loc_rebuilt_2)

            else:
                with torch.no_grad():
                    l_pred, _, attns, l_time_rebuilt, l_loc_rebuilt = model(limages.cuda(), None,None,month,loc_num)
                '''
                    if args.semi_supervise:
                        u_img_rep_1, u_pred_1, u_int_pred_1, u_time_rebuilt_1, u_loc_rebuilt_1 = model(
                                uimages_1.cuda(), u_sl.cuda())
                        u_img_rep_2, u_pred_2, u_int_pred_2, u_time_rebuilt_2, u_loc_rebuilt_2 = model(
                                uimages_2.cuda(), u_sl.cuda())          
                '''

        elif args.model == 'ssnet':
            if train:
                _, l_pred, l_time_rebuilt, l_loc_rebuilt = model(limages.cuda(),month,loc_num)


                if args.semi_supervise:
                    u_img_rep_1, u_pred_1, u_month_out_1, u_loc_out_1= model(uimages_1.cuda(),umonth,uloc_num)
                    u_img_rep_2, u_pred_2, u_month_out_2, u_loc_out_2= model(uimages_2.cuda(),umonth,uloc_num)

                    img_z_1, img_p_1, sem_z_1, sem_p_1 = u_img_rep_1
                    img_z_2, img_p_2, sem_z_2, sem_p_2 = u_img_rep_2

                    # assert torch.sum(torch.isnan(u_img_rep_1)) == 0, print(u_img_rep_1)
                    # assert torch.sum(torch.isnan(u_pred_1)) == 0, print(u_pred_1)
                    # assert torch.sum(torch.isnan(u_int_pred_1)) == 0
                    # assert torch.sum(torch.isnan(u_time_rebuilt_1)) == 0,print(u_time_rebuilt_1)
                    # assert torch.sum(torch.isnan(u_loc_rebuilt_1)) == 0,print(u_loc_rebuilt_1)

                    # assert torch.sum(torch.isnan(u_img_rep_2)) == 0,print(u_img_rep_2)
                    # assert torch.sum(torch.isnan(u_pred_2)) == 0, print(u_pred_2)
                    # assert torch.sum(torch.isnan(u_int_pred_2)) == 0
                    # assert torch.sum(torch.isnan(u_time_rebuilt_2)) == 0,print(u_time_rebuilt_2)
                    # assert torch.sum(torch.isnan(u_loc_rebuilt_2)) == 0,print(u_loc_rebuilt_2)

            else:
                with torch.no_grad():
                    l_img_rep, l_pred, l_time_rebuilt, l_loc_rebuilt = model(limages.cuda(),month,loc_num)
                    '''
                        if args.semi_supervise:
                            u_img_rep_1, u_pred_1, u_int_pred_1, u_time_rebuilt_1, u_loc_rebuilt_1 = model(
                                    uimages_1.cuda(), u_sl.cuda())
                            u_img_rep_2, u_pred_2, u_int_pred_2, u_time_rebuilt_2, u_loc_rebuilt_2 = model(
                                    uimages_2.cuda(), u_sl.cuda())          
                    '''

        else:
            print('model not supported')
            exit(0)



        # supervised loss
        #print('compute the loss')

        # labeled data's supervised loss
        l_loss = F.binary_cross_entropy_with_logits(l_pred.view(labels.size(0),-1),labels.cuda(),reduction='none').sum()/args.batch_size
        #- cosine_dis(l_pred.view(labels.size(0),-1), labels.cuda(), args.batch_size)

        s_loss = torch.tensor(0)
        i_loss = torch.tensor(0)

        if args.semi_supervise and train:
            # sematic loss
            #print(u_pred_1,u_pred_2)

            s_loss = -0.5 * cosine_dis(sem_p_1,sem_z_2,args.batch_size) - 0.5 * cosine_dis(sem_p_2,sem_z_1,args.batch_size)

            '''
            s_loss = - 0.5 * cosine_dis(F.sigmoid(img_p_1.view(u_pred_1.size(0),-1).detach()),
                                  F.sigmoid(u_pred_2.view(u_pred_1.size(0),-1)), args.batch_size)\
                     - 0.5 * cosine_dis(F.sigmoid(u_pred_1.view(u_pred_1.size(0),-1).detach()),
                                  F.sigmoid(u_pred_2.view(u_pred_1.size(0),-1).detach()), args.batch_size)
            '''

            '''
            s_loss = F.binary_cross_entropy(F.sigmoid(u_pred_1.view(u_pred_1.size(0),-1)),
                                            F.sigmoid(u_pred_2.view(u_pred_1.size(0),-1)),reduction='none').mean()*1000/args.batch_size
            '''



            # TODO: batch size may different from labeled data

            # instance loss
            #print(u_img_rep_1.shape,u_img_rep_2.shape)
            #print((u_img_rep_1**2).sum(dim=1).shape,(u_img_rep_2**2).sum(dim=1).shape)
            i_loss = -0.5 * cosine_dis(img_p_1, img_z_2,args.batch_size) - 0.5 * cosine_dis(img_p_2, img_z_1,args.batch_size)

            '''
            i_loss = - 0.5 * cosine_dis(u_img_rep_1, u_img_rep_2.detach(), args.batch_size)\
                     - 0.5 * cosine_dis(u_img_rep_1.detach(),u_img_rep_2,args.batch_size)
            '''

            #print(s_loss,i_loss)
            #F.binary_cross_entropy(F.sigmoid(u_img_rep_1),F.sigmoid(u_img_rep_2),reduction='none').mean()*1000/args.batch_size
            #- cosine_dis(u_img_rep_1, u_img_rep_2, args.batch_size)
            #abs((u_img_rep_1-u_img_rep_2)**2).sum()/args.batch_size # two output seqs

        # rebuilt loss
        time_loss = torch.tensor(0)
        loc_loss = torch.tensor(0)

        if args.use_geo_emb:
            l_time_loss = torch.tensor(0)#torch.sum(((l_sl[:,0].reshape(-1,1).cuda()-F.sigmoid(l_time_rebuilt)))**2)*4/args.batch_size
            l_loc_loss  = torch.tensor(0)#torch.sum((l_sl[:,1:5].reshape(-1,4).cuda()-F.sigmoid(l_loc_rebuilt))**2)/args.batch_size

        if args.model == 'ms_gogo' or args.model == 'ssnet':
            month_all = torch.from_numpy(np.array(range(12))).view(12, 1)
            loc_all = torch.from_numpy(np.array(range(286))).view(286, 1)
            l_month_label_all = torch.zeros(12, 12).scatter_(1, month_all, 1)
            l_loc_label_all = torch.zeros(286, 286).scatter_(1, loc_all, 1)

            if args.use_month:
                month_loss = F.binary_cross_entropy(l_time_rebuilt.view(12, -1), l_month_label_all.cuda(),
                                                    reduction='none').sum() / 12 /args.batch_size
                l_time_loss = month_loss
            if args.use_loc:
                l_loc_loss = F.binary_cross_entropy(l_loc_rebuilt.view(286, -1), l_loc_label_all.cuda(),
                                                  reduction='none').sum() / 286 /args.batch_size

        if args.semi_supervise and train:
            u_time_loss_1 = torch.tensor(0)#torch.sum(((u_sl[:,0].reshape(-1,1).cuda() - u_time_rebuilt_1) * 4) ** 2) / args.batch_size
            u_loc_loss_1 = torch.tensor(0)#torch.sum((u_sl[:,1:5].reshape(-1,4).cuda() - u_loc_rebuilt_1) ** 2) / args.batch_size

            u_time_loss_2 = torch.tensor(0)#torch.sum(((u_sl[:,0].reshape(-1,1).cuda() - u_time_rebuilt_2) * 4) ** 2) / args.batch_size
            u_loc_loss_2 = torch.tensor(0)#torch.sum((u_sl[:,1:5].reshape(-1,4).cuda() - u_loc_rebuilt_2) ** 2) / args.batch_size

            if args.model == 'trans_gogo':
                time_loss = l_time_loss#torch.tensor(0)#(l_time_loss+u_time_loss_1+u_time_loss_2)/3
                loc_loss  = l_loc_loss#torch.tensor(0)#(l_loc_loss + u_loc_loss_1 + u_loc_loss_2)/3

        else:
            u_time_loss_1 = 0
            u_loc_loss_1 = 0
            u_time_loss_2 = 0
            u_loc_loss_2 = 0
            time_loss = torch.tensor(0)
            loc_loss = torch.tensor(0)
            if args.model == 'trans_gogo' or args.model == 'ssnet':
                time_loss = l_time_loss
                loc_loss = l_loc_loss


        loss_out = l_loss +\
                   min((epoch) * 1,1) * 0.001 * s_loss + \
                   min((epoch) * 1,1) * 0.001 * i_loss + \
                   args.lambda_t * time_loss +\
                   args.lambda_l * loc_loss
        #loss_out.require_grad(True)

        #print('loss compute done')


        pbar.set_description("%s: loss:%.2f,sup:%2.2f,sem:%2.2f,ins:%2.2f,time:%2.2f,loc:%2.2f"%(desc,loss_out.item(),l_loss.item(),s_loss.item(),i_loss.item(),time_loss.item(),loc_loss.item()))#

        if train:
            #with torch.autograd.detect_anomaly():
            loss_out.backward()
            # Grad Accumulation
            if ((batch_idx+1)%args.grad_ac_steps == 0):
                optimizer.step()
                optimizer.zero_grad()
                if warmup_scheduler is not None:
                    warmup_scheduler.step()

        #print('update')
        ## Updates ##
        if not train:
            loss_total += loss_out.item()
            unk_loss_total += loss_out.item()
            start_idx,end_idx=(batch_idx*data.batch_size),((batch_idx+1)*data.batch_size)
            loss_total = loss_total / (end_idx - start_idx)

            if l_pred.size(0) != all_predictions[start_idx:end_idx].size(0):
                pred = l_pred.view(labels.size(0),-1)

            #print(pred.shape,end_idx-start_idx)
            all_predictions[start_idx:end_idx] = l_pred.data.cpu()
            all_targets[start_idx:end_idx] = labels.data.cpu()
        batch_idx += 1

    #print('update and return')
    loss_total = loss_total
    unk_loss_total = unk_loss_total

    if not train:
        if ('split_16c' in args.model) or ('mc16' in args.model) or ('together' in args.model):
            return all_predictions,all_targets,all_masks,all_image_ids,loss_total,unk_loss_total,None
        else:
            return all_predictions,all_targets,all_masks,all_image_ids,loss_total,unk_loss_total,None
    else:
        return None, None, None, None, loss_total, unk_loss_total, None




