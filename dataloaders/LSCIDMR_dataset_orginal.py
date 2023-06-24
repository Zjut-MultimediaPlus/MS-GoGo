import csv
import os
import time

import pandas as pd
import numpy as np
import torch.utils.data
import time
import netCDF4
import torchvision.transforms
from skimage import io
from PIL import Image
import cv2

import utils.augmentation
from utils.aug_plus import augmentation_plus_gpu
from utils.aug_plus import augmentation_plus
from dataloaders.data_utils import get_unk_mask_indices,image_loader

def read_rs_to_numpy(in_file):
    with netCDF4.Dataset(in_file) as nf:

        rs_01 = nf.variables["albedo_01"][:].data
        rs_02 = nf.variables["albedo_02"][:].data
        rs_03 = nf.variables["albedo_03"][:].data
        rs_04 = nf.variables["albedo_04"][:].data
        rs_05 = nf.variables["albedo_05"][:].data
        rs_06 = nf.variables["albedo_06"][:].data
        rs_07 = nf.variables["tbb_07"][:].data
        rs_08 = nf.variables["tbb_08"][:].data
        rs_09 = nf.variables["tbb_09"][:].data
        rs_10 = nf.variables["tbb_10"][:].data
        rs_11 = nf.variables["tbb_11"][:].data
        rs_12 = nf.variables["tbb_12"][:].data
        rs_13 = nf.variables["tbb_13"][:].data
        rs_14 = nf.variables["tbb_14"][:].data
        rs_15 = nf.variables["tbb_15"][:].data
        rs_16 = nf.variables["tbb_16"][:].data

    hsi =  np.array((rs_01,rs_02, rs_03, rs_04, rs_05, rs_06, rs_07, rs_08,
                    rs_09,rs_10, rs_11, rs_12, rs_13, rs_14, rs_15, rs_16))
    return hsi

def geo_transform(year,month,day,loc_y,loc_x,L,W,step,patch_l,patch_w,range_l,range_w,start_lat,start_lon):
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
        :return: tuple (year_time, y_result, x_result, l_result, w_result), elements are torch.tensors
    '''

    month_len   = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_prior_normal = [0,  31, 59, 90, 120,151,182,213,244,274,305,335]
    month_prior_leap = [0, 31, 59+1, 90+1, 120+1,151+1,182+1,213+1,244+1,274+1,305+1,335+1]

    leap_mark = year%4 == 0

    prior_day = 0
    year_day  = 0
    year_len  = 0
    #print(year_day.shape)
    if leap_mark:
        prior_day = month_prior_leap[month-1]
        year_len = 366
    else:
        prior_day = month_prior_normal[month-1]
        year_len = 365
    year_day = prior_day + day
    year_time = year_day/year_len
    year_time = torch.Tensor([year_time])

    y = start_lat - ((loc_y*step+0.5*patch_l)/L*range_l)
    x = start_lon + ((loc_x*step+0.5*patch_w)/W*range_w)


    if x>180:
        x = x-360

    y_result = torch.tensor([y - (-90)/180],dtype=torch.float32)
    x_result = torch.tensor([(x - (-180))/360],dtype=torch.float32)
    l_result = torch.tensor([(patch_l/L*range_l)/180],dtype=torch.float32)
    w_result = torch.tensor([(patch_w/W*range_w)/360],dtype=torch.float32)

    output = torch.concat((year_time, y_result, x_result, l_result, w_result))

    return output #(year_time, y_result, x_result, l_result, w_result)

class MLC_Dataset_16c(torch.utils.data.Dataset):


    def __init__(self, csv_file, root_dir, num_labels ,known_labels=0,transform=None,testing=False,tk_ratio=0.25):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.tk_ratio = tk_ratio

        # ground truth
        self.labels_frame = pd.read_csv(csv_file)

        # img dir
        self.root_dir = root_dir

        # transform
        self.transform = transform
        self.testing = testing
        self.num_labels = num_labels
        self.known_labels = known_labels

        # for multiworkers
        self.start = 0
        self.end = len(self.labels_frame) # no need to -1

        # file_path_list
        self.file_path =self.root_dir+'/20190101.npy'
        self.whole_hsi =  np.load(self.file_path)
        self.ymd_str = '20190101'
        self.max = self.whole_hsi.reshape(16,-1).max(axis=1).reshape(16,1,1)
        self.min = self.whole_hsi.reshape(16,-1).min(axis=1).reshape(16,1,1)
        self.whole_hsi = (self.whole_hsi-self.min)/(self.max-self.min)


    def __len__(self):
        # get the length of data
        return len(self.labels_frame)

    def __getitem__(self, idx):
        """

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patch_name = self.labels_frame.iloc[idx, 0].replace('.png', '')
        patch_ymd = patch_name[0:8]
        img_year = int(patch_name[0:4])
        img_month = int(patch_name[4:6])
        img_day = int(patch_name[6:8])
        loc_x, loc_y = int(patch_name[-2:]) - 1, int(patch_name[-5:-3]) - 1

        #print(img_month)

        # update data
        if patch_ymd != self.ymd_str:
            self.ymd_str = patch_ymd
            self.file_path = self.root_dir+'/'+patch_ymd+'.npy'
            self.whole_hsi = np.load(self.file_path)
            self.max = self.whole_hsi.reshape(16, -1).max(axis=1).reshape(16, 1, 1)
            self.min = self.whole_hsi.reshape(16, -1).min(axis=1).reshape(16, 1, 1)
            self.whole_hsi=(self.whole_hsi-self.min)/(self.max-self.min)#*2-1 #normalize to [-1,1]
            arr = self.whole_hsi.reshape((16,-1))
            #print(np.min(arr,axis=1))

        step = 60 #200
        w = 300 #1000
        patch = self.whole_hsi[:, loc_y * step:loc_y * step + w, loc_x * step:loc_x * step + w]
        sl = geo_transform(int(img_year),int(img_month),int(img_day),int(loc_y),int(loc_x),1800,1800,step,w,w,120,120,60,80)

        '''
        #performing while training along with the forwarding of model
        if not self.testing:
            #patch = utils.augmentation.data_augmentation(patch)
            patch = utils.aug_plus.augmentation_plus(patch)
            # print(hsi.shape)
            
        component_start = time.time()
        
        patch = patch*2-1
        '''



        patch = torch.from_numpy(patch.astype('float32'))
        patch = torchvision.transforms.Resize(256)(patch).detach()

        # ----------------------------------------------------------------------------------
        # perform augmentation in gpu
        if not self.testing:
            patch = augmentation_plus_gpu(patch)

        # images = torch.from_numpy(images.astype('float32'))
        #patch = patch * 2 - 1
        # ----------------------------------------------------------------------------------

        component_end = time.time()

        img_loc = [loc_x, loc_y]

        labels = self.labels_frame.iloc[idx, 1:]
        labels = np.array([labels])
        labels = labels.astype('float').reshape((-1))
        # print(labels.shape)
        image_id = self.labels_frame.iloc[idx, 0]


        # hsi = hsi.permute(1, 2, 0)
        # hsi = Image.fromarray(hsi)

        if self.transform:
            # print(hsi.shape)
            #hsi = self.transform(patch)
            labels = torch.Tensor(labels)

        sample = {'image': patch, 'time_and_location': sl, 'labels': labels}
        #occupied_ratio = ((component_end - component_start) / (get_end_time - get_start_time))
        # print(occupied_ratio)
        return sample








