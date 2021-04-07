#encoding:utf-8
# This is the config file

import argparse

def parsers():
    parser = argparse.ArgumentParser("irradiance prediction")
    parser.add_argument('--debug',type=str, default=False,help='whether use debug mode')
    # irradiance value  range 
    parser.add_argument('--irr_min', type=int, default=0, help='min of irradiance')
    parser.add_argument('--irr_max', type=int, default=1500, help='max of irradiance')
    
    # data parameters config
    parser.add_argument('--home_path', type=str, default='/home/zy', help='root path of dataset')
    parser.add_argument('--train_path', type=str,
            default='/home/yyt/zy/dataset/internation/ASI_split_new/train_all', help='path of train dataset')
    parser.add_argument('--test_path', type=str,
            default='/home/yyt/zy/dataset/internation/ASI_split_new/test_order/', help='path of test dataset')
    
    # model parameters config
    parser.add_argument('--model_name', type=str, default='resnet', help='model name that will be used')   
    parser.add_argument('--save_dir', type=str, default='./save', help='save dir of the model')
    parser.add_argument('--save_date', type=str, default='11-6-9:20', help='save date of the model')  
    parser.add_argument('--save_name', type=str, default='11-3-21:31_epoch=500', help='save name of the model')
    parser.add_argument('--save_csv', type=str, default='1', help='save name of the train_val_loss csv file')
    parser.add_argument('--save_predict', type=str, default='./save/prediction.csv', help='save name of the csv file(prediction result in testdataset)')

    parser.add_argument('--epoch', type=int, default=3, help='epoches')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=6, help='number of cpus to train')
    parser.add_argument('--show_fre', type=int, default=200, help='frequency that data will print')
    parser.add_argument('--return_img_id', type=str, default=False, help='whether return the image id')

    # optimizer parameters config
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate of model')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of learning')
    parser.add_argument('--weight_decay', type=float, default=0.9, help='decay of weight')


    # save and load parameters config
    parser.add_argument('--log', type=str, default='./log/', help='save dir of log file')
    parser.add_argument('--result_path', type=str, default='./result/',
            help='save dir of model result')
    parser.add_argument('--fig_path', type=str, default='./figs/', help='save path of figures')

    # cuda parameters
    parser.add_argument('--cuda', type=str, default=True, help='weather use gpu to train')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index to use')
    parser.add_argument('--seed', type=int, default=12345, help='random seed')
    
    # lstm parameters config
    parser.add_argument('--seq_length', type=int, default=1, help='the length of lstm sequence')
    parser.add_argument('--input_size', type=int, default=1, help='the size ofinput vector')
    parser.add_argument('--hid_size', type=int, default=20, help='the size of hiden state')
    parser.add_argument('--layer_num', type=int, default=1, help='number of lstm layers')
    parser.add_argument('--output_size', type=int, default=1, help='number of output size')
    parser.add_argument('--window', type=int, default=5, help='length of window size')
    parser.add_argument('--horizon', type=int, default=1, help='length of horizon size')

    args = parser.parse_args()
    
    return args
    pass
