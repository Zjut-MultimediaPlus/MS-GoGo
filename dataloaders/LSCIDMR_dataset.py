import csv
import os
import random

import pandas as pd
import numpy as np
import torch.utils.data
import torchvision.transforms
from skimage import io
from PIL import Image

import utils.augmentation
from dataloaders.data_utils import get_unk_mask_indices,image_loader

class MLC_Dataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, num_labels ,known_labels=0,transform=None,testing=False,tk_ratio=0.25):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.tk_ratio = tk_ratio

        # 保存图片的索引数据，以方便按照 index 在csv 中读取指定行，然后取出指定图片
        self.landmarks_frame = pd.read_csv(csv_file)
        # 这个只是简单的路径配置，大伙儿也可以自定义更多参数，以完成更多的骚操作
        self.root_dir = root_dir
        # transform 还是很重要的，如果只有一种可能性，**内定**在这里也是无可厚非的
        self.transform = transform
        self.testing = testing
        self.num_labels = num_labels
        self.known_labels = known_labels

    def __len__(self):
        # 这里就不用解释了吧
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        """
        idx ：这里是int类型的，如果batch_size 是 1+ 的话，那就会指定多次此函数
        内部的作用我想就不用多介绍了，相信大家能够看得懂
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        image = image.transpose((2,0,1)) # WHC -> CWH
        #print(image.shape)
        #print(image.shape)

        if not self.testing:
            #image = utils.augmentation.data_augmentation_(image, 255)
            image = utils.augmentation.data_augmentation(image)
        #print(image.shape)

        # convert to PIL img for data augmentation
        #image = image.astype(np.int)
        image = image.transpose((1,2,0))    # CWH -> WHC
        image = Image.fromarray(np.uint8(image))


        img_loc_x = int(img_name[-9:-7])
        img_loc_y = int(img_name[-6:-4])
        img_month = int(img_name[-14:-12]) # 20190101_01_01.png
        img_loc = [img_loc_x,img_loc_y]

        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape((-1))
        #print(landmarks.shape)
        image_id = self.landmarks_frame.iloc[idx, 0]
        sample = {'image': image, 'landmarks': landmarks}



        if self.transform:
            image = self.transform(image)
            labels = torch.Tensor(landmarks)

        #added from this project
        mask = labels.clone()
        tk_ratio = random.uniform(0,0.75)
        unk_mask_indices = get_unk_mask_indices(image, self.testing, self.num_labels, self.known_labels,tk_ratio=tk_ratio)
        mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)
        #print(unk_mask_indices)
        #mask.scatter_(0, torch.Tensor(unk_mask_indices).long(), -1)

        #caption = torch.LongTensor(np.delete(caption,caption[labels==0]))

        sample['image'] = image
        sample['labels'] = labels

        caption = np.array(range(17))
        caption = caption * labels.numpy()
        caption[caption == 0] = 17
        caption.sort()
        caption = torch.LongTensor(caption)
        sample['length'] = labels.sum().to(torch.int64) #for cnn-rnn
        sample['caption'] = caption     # for cnn-rnn/ the index of labels
        #print(sample['length'],sample['caption'])
        sample['mask'] = mask
        sample['imageIDs'] = str(image_id)
        sample['image_loc'] = img_loc
        sample['loc_num'] = (img_loc[0]-1)*11+(img_loc[1]-1)
        sample['month'] = img_month-1
        return sample