import csv
import os
import time

import pandas as pd
import numpy as np
import torch.utils.data
import time
import torchvision.transforms
import lzma
import gzip
from skimage import io
from PIL import Image

import utils.augmentation
from dataloaders.data_utils import get_unk_mask_indices,image_loader

class MLC_Dataset_16c_gz(torch.utils.data.Dataset):

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

    def __len__(self):
        # get the length of data
        return len(self.labels_frame)

    def __getitem__(self, idx):
        """
        """
        get_start_time = time.time()
        #torch.multiprocessing.set_start_method('spawn')
        if torch.is_tensor(idx):
            idx = idx.tolist()

        hsi_name = os.path.join(self.root_dir,
                                self.labels_frame.iloc[idx, 0].replace('.png','.npy.xz'))

 #       now = time.time()

        #hsi = np.load(hsi_name)
        '''
        with gzip.open(hsi_name,'rb') as f:
            buffer = f.read()
            hsi = np.frombuffer(buffer)
        '''

        with lzma.open(hsi_name, 'rb') as f:
            npy_data = f.read()
            hsi = np.frombuffer(npy_data)
        hsi = hsi.reshape((16,256,256))


#        last = time.time()-now

        #hsi = io.imread(hsi_name)

        #hsi = hsi.transpose((1,2,0)) #
        #print (hsi.shape)
        #hsi = Image.fromarray(hsi)


        if not self.testing:
            hsi = utils.augmentation.data_augmentation(hsi)
            # hsi = utils.augmentation.data_augmentation_(hsi, 0.5)
            #print(hsi.shape)
        component_start = time.time()
        hsi = torch.from_numpy(hsi.copy())
        hsi = torchvision.transforms.Resize(256)(hsi)
        component_end = time.time()
        #hsi = hsi.to(dtype=torch.float1)


        img_loc_x = int(hsi_name[-9:-7])
        img_loc_y = int(hsi_name[-6:-4])
        img_month = int(hsi_name[-14:-12])
        img_loc = [img_loc_x,img_loc_y]

        labels = self.labels_frame.iloc[idx, 1:]
        labels = np.array([labels])
        labels = labels.astype('float').reshape((-1))
        #print(labels.shape)
        image_id = self.labels_frame.iloc[idx, 0]
        sample = {'image': hsi, 'labels': labels}

        #hsi = hsi.permute(1, 2, 0)
        #hsi = Image.fromarray(hsi)
        if self.transform:
            #print(hsi.shape)
            #hsi = self.transform(hsi)
            labels = torch.Tensor(labels)

        #added from this project
        mask = labels.clone()
        unk_mask_indices = get_unk_mask_indices(hsi, self.testing, self.num_labels, self.known_labels,self.tk_ratio)
        mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)
        #print(unk_mask_indices)
        #mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)

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
        sample['loc_num'] = (img_loc[0]-1)*11+(img_loc[1]-1)
        sample['month'] = img_month-1
        get_end_time = time.time()
        occupied_ratio = ((component_end-component_start)/(get_end_time-get_start_time))
        #print(occupied_ratio)
        return sample