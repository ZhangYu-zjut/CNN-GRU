# encoding:utf-8
# python=3.6+
import numpy as np
import os
import csv
import pandas as pd

import torch
import torch as t
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

from config import parsers

opt = parsers()
"""
Read the image data
1.read the data
2.split the data
3.do the transform

"""


class DataSet(Dataset):
    # Split the dataset into 3 parts: train,val,test
    """
    File dir:<train>,<test>
    Train set: train=True,test=False
    Val set: train=False,test=False
    Test set: test=True
    """

    def __init__(self, img_path, transform=None, train=True, test=False,
                 train_val_ratio=0.8,normalize_type='mean_std'):
        self.root_path = img_path
        self.test = test
        imgs = [os.path.join(img_path, i) for i in os.listdir(img_path)]
        # df = csv.read(csv_path,header=None,names=[])
        # sort the image by id
        # train: root/train/1.jpg
        # test: root/test/2.jpg
        imgs = sorted(imgs, key=lambda x: int(x.split('/')[-1].split('-')[0]))
        img_num = len(imgs)
        print("img_num", img_num)
        data_mean, data_std = self.get_mean_std(img_path)
        min_d, max_d = self.get_min_max(img_path)
        time_series_data_all = self.get_time_series_data()
        time_series_length = len(time_series_data_all)
        """ 
        Time series data split:
        # all data: [0:len(all_img)]
        # train: [0:len(train_imgs)*train_val_ratio]
        # val: [len(train_imgs)*train_val_ratio:len(train_imgs)]
        # test: [len(all_img) - len(test_imgs) : len(all_img)]
        """
        # normalize first
        time_series_data_all = self.normalize(time_series_data_all,normalize_type)
        self.time_series_data_all_x,self.time_series_data_all_y = self.split_data(time_series_data_all)


        if (self.test):
            self.img = imgs
            #self.time_series_part_x = self.time_series_data_all_x[time_series_length-img_num:time_series_length]
            #self.time_series_part_y = self.time_series_data_all_y[time_series_length - img_num:time_series_length]
        else:
            if train:
                self.img = imgs[:int(train_val_ratio * img_num)]

                #self.time_series_part_x = self.time_series_data_all_x[0:int(img_num*train_val_ratio)]
                #self.time_series_part_y = self.time_series_data_all_y[0:int(img_num * train_val_ratio)]
            else:
                self.img = imgs[int(train_val_ratio * img_num):]
                #self.time_series_part_x = self.time_series_data_all_x[int(img_num*train_val_ratio):img_num]
                #self.time_series_part_y = self.time_series_data_all_y[0:int(img_num * train_val_ratio)]

        # Default transfprm
        if transform == None:
            normalize = T.Normalize(
                mean=[.5, .5, .5],
                std=[.5, .5, .5]
            )
            # for val and test data
            if (test == True or train == False):
                self.transform = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            # for train data
            else:
                self.transform = T.Compose([
                    T.Resize(224),
                    T.RandomCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize])
        pass

    def normalize(self,data,normalize_type):
        if normalize_type == 'mean_std':
            self.mean = np.mean(data)
            self.std = np.std(data)
            data = (data - np.mean(data))/(np.std(data))
        if normalize_type == 'max_min':
            self.max = np.max(data)
            self.min = np.min(data)
            data = (data - np.min(data))/(np.max(data) - np.min(data))
        return data

    def get_max_min(self):
        return self.max, self.min
        pass

    def get_mean_std(self):
        return self.mean, self.std

    def get_time_series_data(self):
        irr_data_all = pd.read_csv(r'irr_data.csv', header=None, names=['irr']).values.reshape(-1, )
        """
        data_train = irr_data_all[:int(opt.train_size * len(irr_data_all))]
        data_test = irr_data_all[int(opt.train_size * len(irr_data_all)):]
        irr_data_all_x, irr_data_all_y = self.split_data(irr_data_all)
        """
        return irr_data_all
        pass


    def split_data(self,data):
        x_data = []
        y_data = []
        window = int(opt.window)
        for i in range(len(data)):
            if (i < window):
                tmp = np.zeros(window)
                tmp[0: (window - i)] = 0
                tmp[(window - i):] = data[:i]
                x_data.append(tmp)
            else:
                tmp = data[(i - window):i]
                x_data.append(tmp)
            y_data.append(data[i])
        x_data = np.array(x_data).astype(float)
        y_data = np.array(y_data).astype(float)
        return x_data, y_data
    # horizon=1, time interval = 10 mins
    def split_data_new(self,data,horizon=1):
        x_data = []
        y_data = []
        window = int(opt.window)
        # i: current position
        for i in range(len(data)-horizon):
            if (i < window):
                tmp = np.zeros(window)
                tmp[0: (window - i)] = 0
                tmp[(window - i):] = data[:i]
                x_data.append(tmp)
            else:
                tmp = data[(i - window):i]
                x_data.append(tmp)
            y_data.append(data[i+horizon])
        x_data = np.array(x_data).astype(float)
        y_data = np.array(y_data).astype(float)
        return x_data, y_data

    def __getitem__(self, index):
        img_path = self.img[index]
        data = Image.open(img_path)
        if (opt.debug):
            print("img_path is", img_path)
        img_data = self.transform(data)
        img_id = int(self.img[index].split('/')[-1].split('-')[0])
        """
        if(img_id>len(self.time_series_part_x)):
            print("len(time_series_part_x)", len(self.time_series_part_x))
            print("img id:", img_id)
        """
        irr_time_series_x = torch.tensor(self.time_series_data_all_x[img_id])
        irr_time_series_y = torch.tensor(self.time_series_data_all_y[img_id])

        target = float(self.img[index].split('/')[-1].split('-')[1])
        target = 1.0 * (target - self.min) / (self.max - self.min)
        target = torch.Tensor(np.array(target))
        if(opt.return_img_id):
            return img_data, irr_time_series_x, irr_time_series_y,img_id
        return img_data,irr_time_series_x,irr_time_series_y
        #return data, target
        pass

    def __len__(self):
        # print("length is",len(self.img))
        return len(self.img)

        pass

    def get_mean_std(self, img_path):
        imgs = [os.path.join(img_path, i) for i in os.listdir(img_path)]
        # print((imgs[0].split('/')[-1]))
        # print((imgs[0].split('/')[-1].split('-')[1]))
        count = 0
        # for x in imgs:
        # print(x.split('/')[-1])
        # print(x.split('/')[-1].split('-')[1])
        # count += 1
        # print("count is",count)
        # print(imgs)
        value = [int(x.split('/')[-1].split('-')[1]) for x in imgs]
        # print("imgs is",imgs)
        # print("values is",value)
        value = np.array(value)
        mean_d, std = np.mean(value), np.std(value)
        # print("mean is ",mean_d,"std is",std)
        self.mean, self.std = mean_d, std

        return mean_d, std
        pass

    def get_min_max(self, img_path):
        imgs = [os.path.join(img_path, i) for i in os.listdir(img_path)]
        value = [int(x.split('/')[-1].split('-')[1]) for x in imgs]
        # print("imgs is",imgs)
        # print("values is",value)
        value = np.array(value)
        min_d, max_d = np.min(value), np.max(value)
        # print("mean is ",mean_d,"std is",std)
        self.min, self.max = min_d, max_d
        return min_d, max_d
        pass
