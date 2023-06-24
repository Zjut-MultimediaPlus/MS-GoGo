
import numpy as np
import logging
from collections import OrderedDict
import torch
import math
from pdb import set_trace as stop
import os
from utils.metrics import *
import torch.nn.functional as F 
import warnings
warnings.filterwarnings("ignore")


class LossLogger():
    def __init__(self,model_name):
        self.model_name = model_name
        open(model_name+'/train.log',"w").close()
        open(model_name+'/valid.log',"w").close()
        open(model_name+'/test.log',"w").close()

    def log_losses(self,file_name,epoch,loss,metrics,train_unk=0):
        log_file = open(self.model_name+'/'+file_name,"a")
        log_file.write(
            '\n\n\nEPOCH:%d\tloss:%.2f\tmAP:%.2f'%(epoch,loss,metrics['mAP']*100))
        log_file.write(
            '\nsubset_ACC:%.2f' % (metrics['ACC'] * 100))
        log_file.write(
            '\nHamming_distance:%.2f\tExample-based_F1:%.2f' % (metrics['HA']*100, metrics['ebF1']*100))
        log_file.write(
            '\nCP:%.2f\tCR:%.2f\tCF1:%.2f' % (metrics['CP']*100, metrics['CR']*100, metrics['CF1']*100))
        log_file.write(
            '\nOP:%.2f\tOR:%.2f\tOF1:%.2f' % (metrics['OP']*100, metrics['OR']*100, metrics['OF1']*100))
        log_file.write(
            '\nCP_t3:%.2f\tCR_t3:%.2f\tCF1_t3:%.2f' % (metrics['CP_top3'] * 100, metrics['CR_top3'] * 100, metrics['CF1_top3'] * 100))
        log_file.write(
            '\nOP_t3:%.2f\tOR_t3:%.2f\tOF1_t3:%.2f' % (metrics['OP_top3'] * 100, metrics['OR_top3'] * 100, metrics['OF1_top3'] * 100))
        log_file.write('\naccs:\nTC:%.2f\tEC:%.2f\tFS:%.2f\tWJ:%.2f\tSnow:%.2f\tOcean:%.2f\tDesert:%.2f\tVegetation:%.2f'
              % tuple(metrics['accuracy'][0:8].data))
        log_file.write('\nCi:%.2f\tCs:%.2f\tDC:%.2f\tAc:%.2f\tAs:%.2f\tNs:%.2f\tCu:%.2f\tSc:%.2f\tSt:%.2f'
              % tuple(metrics['accuracy'].data[8:17]))
        log_file.write(
            '\naccs:\nTC:%.2f\tEC:%.2f\tFS:%.2f\tWJ:%.2f\tSnow:%.2f\tOcean:%.2f\tDesert:%.2f\tVegetation:%.2f'
            % tuple(metrics['APs'][0:8].data))
        log_file.write('\nCi:%.2f\tCs:%.2f\tDC:%.2f\tAc:%.2f\tAs:%.2f\tNs:%.2f\tCu:%.2f\tSc:%.2f\tSt:%.2f'
                       % tuple(metrics['APs'].data[8:17]))

        '''
        str(epoch)+
                       ',  loss:'+str(loss)+
                       '  mAP:'+str(metrics['mAP'])+
                       '\n  mAP:' + str(metrics['mAP']) +
                       ',  sub_ACC:'+str(metrics['ACC'])+'\n'
        '''
        log_file.close()


class Logger():
    def __init__(self,args):
        self.model_name = args.model_name
        self.best_mAP = 0
        self.best_class_acc = 0

        if args.model_name:
            try:
                os.makedirs(args.model_name)
            except OSError as exc:
                pass

            try:
                os.makedirs(args.model_name+'/epochs/')
            except OSError as exc:
                pass

            self.file_names = {}
            self.file_names['train'] = os.path.join(args.model_name,'train_results.csv')
            self.file_names['valid'] = os.path.join(args.model_name,'valid_results.csv')
            self.file_names['test'] = os.path.join(args.model_name,'test_results.csv')

            self.file_names['valid_all_aupr'] = os.path.join(args.model_name,'valid_all_aupr.csv')
            self.file_names['valid_all_auc'] = os.path.join(args.model_name,'valid_all_auc.csv')
            self.file_names['test_all_aupr'] = os.path.join(args.model_name,'test_all_aupr.csv')
            self.file_names['test_all_auc'] = os.path.join(args.model_name,'test_all_auc.csv')
            

            f = open(self.file_names['train'],'w+'); f.close()
            f = open(self.file_names['valid'],'w+'); f.close()
            f = open(self.file_names['test'],'w+'); f.close()
            f = open(self.file_names['valid_all_aupr'],'w+'); f.close()
            f = open(self.file_names['valid_all_auc'],'w+'); f.close()
            f = open(self.file_names['test_all_aupr'],'w+'); f.close()
            f = open(self.file_names['test_all_auc'],'w+'); f.close()
            os.utime(args.model_name,None)
        
        self.best_valid = {'loss':1000000,'mAP':0,'ACC':0,'HA':0,'ebF1':0,'OF1':0,'CF1':0,'meanAUC':0,'medianAUC':0,'meanAUPR':0,'medianAUPR':0,'meanFDR':0,'medianFDR':0,'allAUC':None,'allAUPR':None,
        'concept_acc':0,'class_acc':0}

        self.best_test = {'loss':1000000,'mAP':0,'ACC':0,'HA':0,'ebF1':0,'OF1':0,'CF1':0,'meanAUC':0,'medianAUC':0,'meanAUPR':0,'medianAUPR':0,'meanFDR':0,'medianFDR':0,'allAUC':None,'allAUPR':None,'epoch':0,
        'concept_acc':0,'class_acc':0}


    def evaluate(self,train_metrics,valid_metrics,test_metrics,epoch,model,attens,valid_loss,test_loss, args, file_name):#file_name:the file which use to log best epoch


        if valid_metrics['mAP'] >= self.best_mAP:
            self.best_mAP = valid_metrics['mAP']
            self.best_test['epoch'] = epoch

            for metric in valid_metrics.keys():
                if not 'all' in metric and not 'time'in metric:
                    self.best_valid[metric]= valid_metrics[metric]
                    self.best_test[metric]= test_metrics[metric]

            print('> Saving Model\n')
            save_dict =  {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'valid_mAP': valid_metrics['mAP'],
                'test_mAP': test_metrics['mAP'],
                'valid_loss': valid_loss,
                'test_loss': test_loss
                }
            torch.save(save_dict, args.model_name+'/best_model.pt')

            if attens != None:
                torch.save(attens, args.model_name + '/attens.pt')

        
        print('\n')
        print('**********************************')
        print('best epoch:%d'%self.best_test['epoch'])
        print('best mAP:  {:0.2f}'.format(self.best_test['mAP']*100))
        print('best CF1:  {:0.2f}'.format(self.best_test['CF1']*100))
        print('best OF1:  {:0.2f}'.format(self.best_test['OF1']*100))
        print('best sub_acc:  {:0.2f}'.format(self.best_test['ACC'] * 100))
        print('**********************************')

        log_file = open(self.model_name + '/' + file_name, "a")
        log_file.write(
            '\nbest_epoch:%d' % self.best_test['epoch'])
        log_file.write(
            '\n********************************************************************')

        return self.best_valid,self.best_test
