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
from scipy.stats import beta
from PIL import Image
import cv2

import utils.augmentation
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
        x = x -360
    #x[x>180] = x[x>180] - 360

    y_result = torch.tensor([y - (-90)/180],dtype=torch.float32)
    x_result = torch.tensor([(x - (-180))/360],dtype=torch.float32)
    l_result = torch.tensor([(patch_l/L*range_l)/180],dtype=torch.float32)
    w_result = torch.tensor([(patch_w/W*range_w)/360],dtype=torch.float32)

    output = torch.concat((year_time, y_result, x_result, l_result, w_result), dim=0)

    return output#(year_time, y_result, x_result, l_result, w_result)


class Unlabel_Dataset_16c_gpu(torch.utils.data.Dataset):


    def __init__(self, csv_file, root_dir, testing=False, device=torch.device('cuda:0')):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # device gpu
        self.device = device

        # ground truth
        self.data_frame = pd.read_csv(csv_file)

        # img dir
        self.root_dir = root_dir

        # transform
        self.testing = testing

        # for multiworkers
        self.start = 0
        #print(self.labels_frame)
        self.end = len(self.data_frame) # no need to -1

        # file_path_list
        random_flag_1 = np.random.randint(low=0,high=5,size=1)
        self.file_path_1 =self.root_dir+'/2018_00%1d0/NC_H08_20180101_00%1d0_R21_FLDK.02401_02401.nc' % (random_flag_1,random_flag_1)
        self.whole_hsi_1 =  read_rs_to_numpy(self.file_path_1)
        self.whole_hsi_1 = torch.from_numpy(self.whole_hsi_1).to(self.device)

        random_flag_2 = np.random.randint(low=0,high=5,size=1)
        self.file_path_2 = self.root_dir + '/2018_00%1d0/NC_H08_20180101_00%1d0_R21_FLDK.02401_02401.nc' % (random_flag_2, random_flag_2)
        self.whole_hsi_2 = read_rs_to_numpy(self.file_path_2)
        self.whole_hsi_2 = torch.from_numpy(self.whole_hsi_2).to(self.device)

        self.year_str = '2018'
        self.month_str = '01'
        self.day_str = '01'
        self.ymd = '20180101'

        self.max_1,_ = torch.max(self.whole_hsi_1.reshape(16,-1),dim=1)#.reshape(16,1,1)
        self.max_1 = self.max_1.reshape(16,1,1)
        self.min_1, _ = torch.min(self.whole_hsi_1.reshape(16, -1), dim=1)  # .reshape(16,1,1)
        self.min_1 = self.min_1.reshape(16, 1, 1)
        self.whole_hsi_1 = (self.whole_hsi_1-self.min_1)/(self.max_1-self.min_1)

        self.max_2, _ = torch.max(self.whole_hsi_2.reshape(16, -1), dim=1)  # .reshape(16,1,1)
        self.max_2 = self.max_2.reshape(16, 1, 1)
        self.min_2, _ = torch.min(self.whole_hsi_2.reshape(16, -1), dim=1)  # .reshape(16,1,1)
        self.min_2 = self.min_2.reshape(16, 1, 1)
        self.whole_hsi_2 = (self.whole_hsi_2 - self.min_2) / (self.max_2 - self.min_2)


    def __len__(self):
        # get the length of data
        return len(self.data_frame)

    def __getitem__(self, idx):
        """

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            sample_year = int(self.data_frame.iloc[idx, 0])
            sample_month = int(self.data_frame.iloc[idx, 1])
            sample_day = int(self.data_frame.iloc[idx, 2])
            sample_y = int(self.data_frame.iloc[idx, 3]) - 1
            sample_x = int(self.data_frame.iloc[idx, 4]) - 1
            sample_ymd = '%04d%02d%02d'%(sample_year,sample_month,sample_day)
            #print( self.data_frame.iloc[idx, :])
        except Exception:
            print(self.data_frame.iloc[idx, :])


        #print(img_month)

        # update data
        if sample_ymd != self.ymd:
            #print(self.ymd, sample_ymd, self.ymd != sample_ymd)
            self.ymd = sample_ymd

            # get first moment
            random_flag_1 = np.random.randint(low=0,high=5,size=1)
            self.file_path_1 = self.root_dir + '/%4d_00%01d0/NC_H08_%04d%02d%02d_00%01d0_R21_FLDK.02401_02401.nc' % (
                sample_year, random_flag_1,sample_year,sample_month,sample_day, random_flag_1)

            while not os.path.exists(self.file_path_1):
                #print('random_select',sample_ymd)
                random_flag_1 = np.random.randint(low=0,high=5,size=1)#(random_flag_1+1)%5
                self.file_path_1 = self.root_dir + '/%4d_00%01d0/NC_H08_%04d%02d%02d_00%01d0_R21_FLDK.02401_02401.nc' % (
                    sample_year, random_flag_1,sample_year, sample_month, sample_day, random_flag_1)



            self.whole_hsi_1 = read_rs_to_numpy(self.file_path_1)
            self.whole_hsi_1 = torch.from_numpy(self.whole_hsi_1).to(self.device)
            self.max_1, _ = torch.max(self.whole_hsi_1.reshape(16, -1), dim=1)  # .reshape(16,1,1)
            self.max_1 = self.max_1.reshape(16, 1, 1)
            self.min_1, _ = torch.min(self.whole_hsi_1.reshape(16, -1), dim=1)  # .reshape(16,1,1)
            self.min_1 = self.min_1.reshape(16, 1, 1)
            self.whole_hsi_1=(self.whole_hsi_1-self.min_1)/(self.max_1-self.min_1) #normalize to [0,1]

            # get second moment
            random_flag_2 = np.random.randint(low=0,high=5,size=1)
            self.file_path_2 = self.root_dir + '/%4d_00%1d0/NC_H08_%04d%02d%02d_00%1d0_R21_FLDK.02401_02401.nc' % (
                sample_year, random_flag_2, sample_year,sample_month, sample_day, random_flag_2)
            while not os.path.exists(self.file_path_2):
                #print('random_select', sample_ymd)
                random_flag_2 = np.random.randint(low=0,high=5,size=1)#(random_flag_2+1)%5
                self.file_path_2 = self.root_dir + '/%4d_00%1d0/NC_H08_%04d%02d%02d_00%1d0_R21_FLDK.02401_02401.nc' % (
                    sample_year, random_flag_2,sample_year, sample_month, sample_day, random_flag_2)


            self.whole_hsi_2 = read_rs_to_numpy(self.file_path_2)
            self.whole_hsi_2 = torch.from_numpy(self.whole_hsi_2).to(self.device)
            self.max_2, _ = torch.max(self.whole_hsi_2.reshape(16, -1), dim=1)  # .reshape(16,1,1)
            self.max_2 = self.max_2.reshape(16, 1, 1)
            self.min_2, _ = torch.min(self.whole_hsi_2.reshape(16, -1), dim=1)  # .reshape(16,1,1)
            self.min_2 = self.min_2.reshape(16, 1, 1)
            self.whole_hsi_2 = (self.whole_hsi_2 - self.min_2) / (self.max_2 - self.min_2)  # normalize to [0,1]


        step = 80 #200
        w = 400 #1000
        random_range = 100
        w = int(beta.rvs(2,2,size=1)[0]*random_range+w-random_range/2)


        shift_y = int(np.random.uniform(0,80,size=None))
        y_end = min(2401, sample_y * step+w + shift_y)
        shift_x = int(np.random.uniform(0,80,size=None))
        x_end = min(2401, sample_x * step+w + shift_x)


        sample_1 = self.whole_hsi_1[:, y_end - w:y_end, x_end - w:x_end].copy()
        sample_2 = self.whole_hsi_2[:, y_end - w:y_end, x_end - w:x_end].copy()



        # get normalized location box and time
        time_and_loc = geo_transform(sample_year,sample_month,sample_day,sample_y,sample_x,2401,2401,step,w,w,120,120,60,80)

        if not self.testing:
            if np.min(sample_1.min()<0):
                print(sample_1.min())

            #sample_1 = utils.augmentation.data_augmentation(sample_1)
            sample_1 = utils.aug_plus.augmentation_plus_gpu(sample_1.copy())#.copy()
            sample_2 = utils.aug_plus.augmentation_plus_gpu(sample_2.copy())#.copy()
            #
            # print(hsi.shape)

        sample_1 = sample_1 * 2 - 1  # normalize ot [-1,1]
        sample_2 = sample_2 * 2 - 1

        component_start = time.time()
        sample_1 = torch.from_numpy(sample_1.astype('float32'))
        sample_1 = torchvision.transforms.Resize(256)(sample_1)
        sample_2 = torch.from_numpy(sample_2.astype('float32'))
        sample_2 = torchvision.transforms.Resize(256)(sample_2)
        component_end = time.time()

        out_version_1 = {'image': sample_1}
        out_version_2 = {'image': sample_2}

        out_version_1['loc_num'] = (img_loc[0]) * 11 + (img_loc[1])
        # print(img_loc,sample['loc_num'])
        out_version_1['month'] = img_month


        return out_version_1,out_version_2








