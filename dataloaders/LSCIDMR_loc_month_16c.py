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
import utils

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

    hsi = np.array((rs_01, rs_02, rs_03, rs_04, rs_05, rs_06, rs_07, rs_08,
                    rs_09, rs_10, rs_11, rs_12, rs_13, rs_14, rs_15, rs_16))
    return hsi


class MLC_Dataset_16c_with_loc_and_month(torch.utils.data.Dataset):


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
        self.file_path = self.root_dir + '/NC_H08_20190101_0020_R21_FLDK.02401_02401.nc'

        #self.whole_hsi =  np.load(self.file_path)
        self.whole_hsi = read_rs_to_numpy(self.file_path)
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
        loc_x,loc_y = int(patch_name[-2:])-1,int(patch_name[-5:-3])-1
        img_month = int(patch_name[4:6])-1
        #print(img_month)

        # update data
        if patch_ymd != self.ymd_str:
            self.ymd_str = patch_ymd
            self.file_path = self.root_dir+'/'+patch_ymd+'.npy'
            self.file_path = self.root_dir + '/NC_H08_%s_0020_R21_FLDK.02401_02401.nc' % patch_ymd
            #self.whole_hsi = np.load(self.file_path)
            self.whole_hsi = read_rs_to_numpy(self.file_path)
            self.max = self.whole_hsi.reshape(16, -1).max(axis=1).reshape(16, 1, 1)
            self.min = self.whole_hsi.reshape(16, -1).min(axis=1).reshape(16, 1, 1)
            self.whole_hsi=(self.whole_hsi-self.min)/(self.max-self.min)#*2-1
            arr = self.whole_hsi.reshape((16,-1))
            #print(np.min(arr,axis=1))

        step = 80#60 #200
        w = 400#300 #1000
        patch = self.whole_hsi[:, loc_y * step:loc_y * step + w, loc_x * step:loc_x * step + w]

        #patch = patch/m.reshape((16,1,1))
        #print(patch.shape)

        #       now = time.time()



        #        last = time.time()-now

        # hsi = io.imread(hsi_name)

        # hsi = hsi.transpose((1,2,0)) #
        # print (hsi.shape)
        # hsi = Image.fromarray(hsi)



        component_start = time.time()
        patch = torch.from_numpy(patch.copy())
        patch = torchvision.transforms.Resize(256)(patch)

        if not self.testing:
            patch = utils.aug_plus.augmentation_plus_gpu(patch)
            # print(hsi.shape)
        patch = patch*2-1
        component_end = time.time()
        # hsi = hsi.to(dtype=torch.float1)


        img_loc = [loc_x, loc_y]

        labels = self.labels_frame.iloc[idx, 1:]
        labels = np.array([labels])
        labels = labels.astype('float').reshape((-1))
        # print(labels.shape)
        image_id = self.labels_frame.iloc[idx, 0]
        sample = {'image': patch, 'labels': labels}

        # hsi = hsi.permute(1, 2, 0)
        # hsi = Image.fromarray(hsi)
        if self.transform:
            # print(hsi.shape)
            hsi = self.transform(patch)
            labels = torch.Tensor(labels)

        # added from this project
        mask = labels.clone()
        #unk_mask_indices = get_unk_mask_indices(hsi, self.testing, self.num_labels, self.known_labels, self.tk_ratio)
        #mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)
        # print(unk_mask_indices)
        # mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)

        caption = np.array(range(17))
        caption = caption * labels.numpy()
        caption[caption == 0] = 17
        caption.sort()
        caption = torch.LongTensor(caption)
        sample['length'] = labels.sum().to(torch.int64)  # for cnn-rnn
        sample['caption'] = caption  # for cnn-rnn/ the index of labels

        sample['image'] = hsi
        sample['labels'] = labels
        sample['mask'] = mask
        sample['imageIDs'] = str(image_id)
        sample['image_loc'] = img_loc

        sample['loc_num'] = (img_loc[0]) * 11 + (img_loc[1])
        #print(img_loc,sample['loc_num'])
        sample['month'] = img_month
        get_end_time = time.time()
        #occupied_ratio = ((component_end - component_start) / (get_end_time - get_start_time))
        # print(occupied_ratio)
        return sample