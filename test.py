# encoding:utf-8
# This is the main file.
"""
Author: yu,Zhang
Time: 2020/09/16
"""
import time
import numpy as np
import os

import torch
import torch as t
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models import resnet18
import torch.nn.functional as F

from config import parsers
opt = parsers()
from model.resnet34 import ResNet34
from model.resnet18 import ResNet18
from dataset import DataSet 
import visdom
import ipdb
from myutils import Flatten
import matplotlib.pyplot as plt
import pandas as pd
from math import *
from model.CNNGRU import CNNGRU

from sklearn.metrics import mean_squared_error
def RMSE(y, y_hat):
    return np.sqrt(mean_squared_error(y, y_hat))

@torch.no_grad()
def evaluate(gt, predict):
    print("len is", len(gt))
    l1_sum = 0
    for i in range(len(gt)):
        temp = abs(gt[i] - predict[i])
        # print("i is, temp is",i,temp)
        l1_sum += abs(gt[i] - predict[i])
    print("l1_sum is", l1_sum)

    mae = 1.0 * np.sum(np.abs(gt - predict)) / len(gt)
    rmse = 1.0 * sqrt(np.sum((gt - predict) * (gt - predict)) / len(gt))
    target_all = gt
    mean_p = predict.mean()
    mean_g = target_all.mean()
    sigma_p = predict.std()
    sigma_g = target_all.std()
    # mse = 1.0 * total_loss_L2/total_sample

    correlation = ((predict - mean_p) * (target_all - mean_g)).mean(axis=0) / (sigma_p * sigma_g)

    return mae, rmse, correlation

def test(model,dataloader,dataset,opt):
    predict_list = []
    gt_list = []

    predict_array = None
    gt_array = None
    for ii,(data, irr_x, target) in enumerate(dataloader):
        # original is data.cuda()
        irr_x = irr_x.cuda()
        data = data.cuda()
        target = target.cuda()
        output = model(data,irr_x )
        target = target.reshape(-1,1)
        test_min = dataset.min
        test_max = dataset.max
        #print("min is",test_min)
        #print("max is",test_max)
        output_numpy = output.cpu().data.numpy()
        gt_numpy = target.cpu().data.numpy()

        output_numpy = (output_numpy * dataset.std + dataset.mean)
        gt_numpy = gt_numpy * dataset.std + dataset.mean


        output_numpy = np.array(output_numpy, dtype=int).squeeze()
        gt_numpy = np.array(gt_numpy, dtype=int).squeeze()

        predict_list.append(output_numpy)
        gt_list.append(gt_numpy)
        if predict_array is None:
            predict_array = output_numpy
            gt_array = gt_numpy
        else:
            predict_array = np.concatenate((predict_array,output_numpy))
            gt_array = np.concatenate((gt_array, gt_numpy))
    #predict_array = np.array(predict_list).reshape(-1,1200)
    print("predict array is:",predict_array)
    print(len(predict_array))
    print("actuall array is:",gt_array)
    print(len(gt_array))
    mae, rmse, corr = evaluate(gt_array, predict_array)
    print("mae is: %.5f, rmse is: %.5f corr is: %.5f" % (mae, rmse, corr))
    save_path = './save/%s/%s.csv' % (opt.save_date, opt.save_predict)
    save_data(predict_array, save_path)

    save_data2(gt_array, predict_array, save_path)
    y = gt_array
    x = predict_array
    search(x, y)
    pass

def save_data(data,save_path):
    df = pd.DataFrame(data=data)
    df.to_csv(save_path,sep=',',index=None,header=None)
    pass
def save_data2(data1,data2,save_path):
    df = pd.DataFrame(data=[data1,data2]).T
    df.to_csv(save_path,sep=',',index=None,header=None)
    pass
# visualize the predict result
def predict_vis(predict, gd, time_list):
    
    plt.figure()
    plt.plot()

    plt.xlabel("time")
    plt.ylabel("value")
    plt.title("predict and actuall comparison")
    plt.savefig(opt.fig_save_path+"compare_result.jpg",dpi=300)
    plt.show()
    pass

# visualize the loss and MSE error.
def visualize(epoch_list,loss_list):
    vis = visdom.Visdom(env='loss curve')
    vis.line(X=epoch_list, Y=loss_list, win='loss',
            opts={'title':'loss curve', 'xlabel':'epoch', 'ylabel':'loss'})
    pass

multiple = 0.01
gain = 0.01
epochs = 1000
col = 'res152'

def search(x, y):
    global multiple, result
    result = np.zeros([1, len(y)], dtype=np.float32)
    rmse, multi = 1e15, 0
    for epoch in range(epochs):
        y_hat = x * multiple
        tmpRmse = RMSE(y, y_hat)
        if rmse > tmpRmse:
            rmse = tmpRmse
            multi = multiple
            result = y_hat
        multiple += gain

    print("new rmse is:", rmse)
    with open('./best.csv', 'w') as f:
        for num in list(result):
            f.write(str(num) + '\n')


if __name__ == "__main__":
    device = torch.device('cuda')
    torch.manual_seed(1234)

    # input_dim,hidden_dim,batch_size,cnn_output_dim=1,output_dim=1,layer_num=2)
    model = CNNGRU(cnn_output_dim=1, gru_seq_length=opt.window, gru_input_dim=1, gru_hidden_dim=10,
                   gru_batch_size=opt.batch_size, gru_output_dim=1, gru_layer_num=2)
    model = model.cuda()

    optimizer = optim.SGD(model.parameters(),lr=opt.lr,momentum=opt.momentum)

    test_data = DataSet(opt.test_path,train=False,test=True)
    test_loader = DataLoader(test_data,
                            batch_size = opt.batch_size,
                            num_workers = opt.num_workers)
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    if torch.cuda.is_available():
        if opt.cuda == False:
            print("Warning: You have GPU but you did't set to GPU train!")
        else:
            model = model.cuda()
    torch.manual_seed(opt.seed)

    # Test the model generation in test dataset
    model_path ="%s/%s/%s_epoch=%s.pt"%(opt.save_dir,opt.save_date,opt.save_name,opt.epoch)
    
    with open(model_path,'rb') as f:
        model.load_state_dict(torch.load(f))
    #test_loss, test_rae, test_rse, test_corr = evaluate(test_loader, model,evaluateL1, evaluateL2)
    test(model,test_loader,test_data, opt)


