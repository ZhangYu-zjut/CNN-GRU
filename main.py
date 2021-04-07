# encoding:utf-8
# This is the main file.
"""
Author: yu,Zhang
Time: 2020/09/16
"""
import time
import numpy as np
import os
import pandas as pd

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
from model.CNNGRU import CNNGRU

opt = parsers()
from model.resnet34 import ResNet34
from model.resnet18 import ResNet18
from dataset import DataSet
import visdom
import ipdb
from myutils import Flatten
from matplotlib import pyplot as plt

loss_item = []
epoch_list = []
loss_list = []


def weight_init(m):
    classname = m.__class__.__name__  # 2
    if classname.find('Conv') != -1:  # 3
        nn.init.normal_(m.weight.data, 0.0, 0.02)  # 4
    elif classname.find('BatchNorm') != -1:  # 5
        nn.init.normal_(m.weight.data, 1.0, 0.02)  # 6
        nn.init.constant_(m.bias.data, 0)
        # nn.init.normal_(m.weight.data,0.0, 0.02)


def train(dataloader, model, optimizer, criterion):
    model.train()
    total_loss = 0
    n_samples = 0
    # criterion = nn.MSELoss()
    count = 0
    """
    data shape:
    data: [N,C,H,W]
    irr_x: [B,window]
    irr_y: [B,1]
    """
    for j, (data, irr_x, irr_y) in enumerate(dataloader):
        target = irr_y
        if (j == 0):
            # print("data is",data)
            # print("data size is",data.size())
            pass
            # print("target.size(1) is",target.size(1))
        if (opt.cuda):
            irr_x = irr_x.cuda()
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        # with torch.no_grad():

        target = target.reshape(-1, 1)
        # ipdb.set_trace()
        #print("irr_x.size()",irr_x.size())
        predict = model(data,irr_x)
        target = target.to(torch.float32)
        if (j < 2):
            if (opt.debug):
                print("j is,", j, " predict is", predict.data)
                print("predict",predict)
                print("target",target)
            pass

        loss = criterion(predict, target)
        # print("data is",data)
        # print("predict is",predict)
        # print("loss is",loss)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_samples += target.size(0)
        count += 1
    loss_item.append(total_loss)
    # val_loss, val_rae, val_rse = evaluate(val_loader, model,evaluateL1, evaluateL2)
    # print("loss is",loss.item())
    return 1.0 * total_loss / n_samples
    pass


# MSE\MAE\Correlation
@torch.no_grad()
def evaluate(dataloader, model, evaluateL1, evaluateL2):
    model.eval()
    total_loss_L1 = 0
    total_loss_L2 = 0
    total_sample = 0
    total_target = 0
    predict = None
    target_all = None

    for ii, (data, irr_data_x,target) in enumerate(dataloader):
        data = data.cuda()
        target = target.cuda()
        irr_data_x = irr_data_x.cuda()
        """
        try:
            output = model(data)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception
        """
        output = model(data,irr_data_x)
        target = target.reshape(-1, 1)
        # concatnate the predict and target
        if predict is None:
            # print("predict size is",predict.size())
            predict = output
            target_all = target

        else:
            predict = torch.cat((predict, output))
            target_all = torch.cat((target_all, target))
            pass
        # print("output size",output.size(),"target size:",target.size())
        total_loss_L1 += evaluateL1(output, target).item()
        total_loss_L2 += evaluateL2(output, target).item()
        total_sample += opt.batch_size
        pass

    mean_p = predict.mean()
    mean_g = target_all.mean()
    sigma_p = predict.std()
    sigma_g = target_all.std()
    mae = 1.0 * total_loss_L1 / total_sample
    mse = 1.0 * total_loss_L2 / total_sample
    rae = 1.0 * total_loss_L1 / t.sum(t.abs(target_all.mean() - target_all))
    rse = 1.0 * total_loss_L2 / t.sum((target_all.mean() - target_all) ** 2)
    correlation = ((predict - mean_p) * (target_all - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
    # correlation = (mean_p * mean_g).mean()/(sigma_p *sigma_g)
    index = (sigma_g != 0)
    correlation = (correlation[index]).mean()
    return total_loss_L2, rae, rse, correlation
    pass


# visualize the predict result
def predict_vis(dataloader, dataset):
    predict_list = []
    gt_list = []
    for ii, (data, irr_data_x,target) in enumerate(dataloader):
        # original is data.cuda()
        data = data.cuda()
        irr_data_x = irr_data_x.cuda()
        target = target.cuda()
        output = model(data,irr_data_x)
        target = target.reshape(-1, 1)
        """
        # concatnate the predict and target
        if predict is None:                                                   
            #print("predict size is",predict.size())
            predict = output
            target_all = target
        else:
            predict = torch.cat((predict, output))
            target_all = torch.cat((target_all, target))
        pass
        """
        test_min = dataset.min
        test_max = dataset.max

        # print("min is",test_min)
        # print("max is",test_max)
        output_numpy = output.cpu().data.numpy()
        gt_numpy = target.cpu().data.numpy()

        output_numpy = (output_numpy * dataset.std + dataset.mean)
        gt_numpy = gt_numpy * dataset.std + dataset.mean
        predict_list.append(output_numpy)
        gt_list.append(gt_numpy)
        pass
    # 37*32+16=1200
    # print("predict:",predict_list)
    # print("ground truth:",gt_list)
    p_len = len(predict_list[0])  # 32
    g_len = len(gt_list[0])  # 32
    # print(p_len)
    predict_list = np.array(predict_list[0]).squeeze()
    gt_list = np.array(gt_list[0]).squeeze()

    # plt.figure()
    # x:(32,) predict(32,)
    x = np.array([i for i in range(opt.batch_size)])
    fig, ax = plt.subplots()
    predict_line, = ax.plot(x, predict_list, label='predict value')
    gt_line, = ax.plot(x, gt_list, label='ground truth value')
    ax.legend()
    ax.grid(True)
    # plt.plot(predict_list,gt_list)

    plt.xlabel("time")
    plt.ylabel("value")
    plt.title("predict and actuall comparison")
    plt.savefig(opt.fig_path + "compare_result_epcoch=%d.svg" % opt.epoch, format='svg', dpi=1200)
    plt.show()
    pass

    # visualize the loss and MSE error.
    def visualize(epoch_list, loss_list):
        vis = visdom.Visdom(env='loss curve')
        vis.line(X=epoch_list, Y=loss_list, win='loss',
                 opts={'title': 'loss curve', 'xlabel': 'epoch', 'ylabel': 'loss'})
        pass

    # save the train and val loss data


def save(data):
    # col_name = ["train_loss","val_loss"]
    # columns=col_name
    data = np.array(data).T
    df = pd.DataFrame(data=data)
    save_path = "./save/%s/train_val_loss_%s_epoch=%s.csv" % (opt.save_date, opt.save_csv, opt.epoch)
    df.to_csv(save_path, encoding='utf-8', index=False, header=None)
    print("save loss data!")


if __name__ == "__main__":
    save_result_path = "./save/%s/" % (opt.save_date)
    if (os.path.exists(save_result_path) == False):
        os.mkdir(save_result_path)
    # model = tv.models.resnet18(pretrained=True)
    # model.fc = nn.Linear(2048,1)
    # model = ResNet18(num_classes=1)
    # model.apply(weight_init)
    #import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    device = torch.device('cuda')
    torch.manual_seed(1234)

    # input_dim,hidden_dim,batch_size,cnn_output_dim=1,output_dim=1,layer_num=2)
    model = CNNGRU(cnn_output_dim=1,gru_seq_length=opt.window,gru_input_dim=1,gru_hidden_dim=10,
                 gru_batch_size=opt.batch_size,gru_output_dim=1,gru_layer_num=2)
    model = model.cuda('cuda:'+str(opt.gpu))

    # print("model is",model)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # add the scheduler which is used for learning rate adjusting.
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # Get the data
    # define the model
    # Define the other parameters
    # train the model
    # evaluate and test the model
    train_data = DataSet(opt.train_path, train=True, test=False)
    val_data = DataSet(opt.train_path, train=False, test=False)
    test_data = DataSet(opt.test_path, train=False, test=True)
    train_loader = DataLoader(train_data,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers)

    val_loader = DataLoader(val_data,
                            batch_size=opt.batch_size,
                            num_workers=opt.num_workers)

    test_loader = DataLoader(test_data,
                             batch_size=opt.batch_size,
                             num_workers=opt.num_workers)

    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # inputs = inputs.to(device)
    if torch.cuda.is_available():
        if opt.cuda == False:
            print("Warning: You have GPU but you did't set to GPU train!")
        else:
            # train_loader = train_loader.cuda()
            # val_loader = val_loader.cuda()
            # test_loader = test_loader.cuda()
            # optimizer = optimizer.cuda()
            model = model.cuda()

    torch.manual_seed(opt.seed)
    best_val = 1000000000

    criterion = nn.MSELoss(reduction='sum')
    evaluateL1 = nn.L1Loss(reduction='sum')
    evaluateL2 = nn.MSELoss(reduction='sum')
    train_loss_list = []
    val_loss_list = []
    row1_list = []
    row2_list = []
    loss_list = []
    if opt.cuda:
        criterion = criterion.cuda()
        evaluateL1 = evaluateL1.cuda()
        evaluateL2 = evaluateL2.cuda()
    try:
        for epoch in range(opt.epoch):

            epoch_st_time = time.time()
            train_loss = train(train_loader, model, optimizer, criterion)
            val_loss, val_rae, val_rse, val_corr = \
                evaluate(val_loader, model, evaluateL1, evaluateL2)
            # scheduler.step()
            print(
                "epoch: {:3d} | time: {:2.2f}s | train loss {:2.8f} | val_rae: {:2.5f} | val_rse: {:2.5f} | val_corr: {:5.4f}" \
                .format(epoch, time.time() - epoch_st_time, train_loss, \
                        val_rae, val_rse, val_corr))
            if val_loss < best_val:
                best_val = val_loss
                model_path = "%s/%s/%s_epoch=%s.pt" % (opt.save_dir, opt.save_date, opt.save_name, opt.epoch)
                with open(model_path, 'wb') as f:
                    torch.save(model.state_dict(), f)
            epoch_list.append(epoch)
            row1_list.append(train_loss)
            row2_list.append(val_loss)
        loss_list.append(row1_list)
        loss_list.append(row2_list)
        # train_loss_list.append(train_loss)
        # val_loss_list.append(val_loss)
        save(loss_list)
    # train(model,train_loader,optimizer, criterion)
    except KeyboardInterrupt:
        print("-" * 90)
        print("Exiting from training early!!!")

    # Test the model generation in test dataset
    model_path = "%s/%s/%s_epoch=%s.pt" % (opt.save_dir, opt.save_date, opt.save_name, opt.epoch)
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    test_loss, test_rae, test_rse, test_corr = evaluate(test_loader, model, evaluateL1, evaluateL2)

    # print("test data min is{:4d}, max is{:4d}".format(test_data.min,test_data.max))
    print("test loss: {:5.2f} | test rae: {:5.2f} | test rse: {:5.2f} | test corr:{:5.4f}".format(test_loss, test_rae,
                                                                                                  test_rse, test_corr))
    predict_vis(test_loader, test_data)

