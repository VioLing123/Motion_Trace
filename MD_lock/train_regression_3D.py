import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
import logging
import sys
import os
import dataset.dataset as dataset
from tqdm import tqdm
from datetime import datetime
import random
import numpy as np
from tensorboardX import SummaryWriter
# self-define
from net.net import *
from Det_single_label.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--mode',type=str,
                    default='Regression', help='Classify or Regression')
parser.add_argument('--net',type=str,
                    default='SR',
                    help='CR=>Conv_Net_Residual, SR=>Simple_Residual, AN=>Activation_Net, CN=Conv_Net, CRN = Conv_Net_Residual_New')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  
                    default=0.001,help='segmentation network learning rate')#学习率
parser.add_argument('--img_size', type=int,
                    default=128, help='input patch size of network input')#输入图像的大小
parser.add_argument('--data_root', type=str,
                    default='./data/speed_3D_t', help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=3407, help='random seed') # former 3407
parser.add_argument('--loss',type=str,
                    default='SmoothL1', help='loss_function')
parser.add_argument('--withReset', type=int,
                    default= 1, help='device feature')
parser.add_argument('--Resume', type=int,
                    default=0, help='train from checkpoint')
args = parser.parse_args()

if __name__ == "__main__":

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # 器件加擦除的数据集
    if args.withReset:
        list_path_train = f'{args.data_root}/R_train_3D_25/train.csv'
        list_path_eval = f'{args.data_root}/R_eval_3D_25/eval.csv'
    else:
        # 器件加擦除的数据集
        list_path_train = f'{args.data_root}/R_train_noRes/train.csv'
        list_path_eval = f'{args.data_root}/R_eval_noRes/eval.csv'

    # 产对应文件夹
    log_path=f'./log/{args.mode}/{args.net}'
    if not os.path.isdir(log_path):
        os.makedirs(log_path)

    tensorboard_path = f'./run/{args.mode}/{args.net}'
    if not os.path.isdir(tensorboard_path):
        os.makedirs(tensorboard_path)

    model_save_path = f'./save/{args.mode}/{args.net}'
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    #时间戳
    TIMESTAMP = "{0:%m-%d-%H-%M}".format(datetime.now())
    
    path = f'{TIMESTAMP}_{args.net}_{args.max_epochs}eps_{args.base_lr}lr'
    #打log
    logging.basicConfig(filename=f'{log_path}/{path}.csv',level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s ', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    writer = SummaryWriter(f'{tensorboard_path}/{path}')    
    
    basic_info = f'model:{args.mode}\nnet:{args.net},withReset:{args.withReset}\nmax_epochs:{args.max_epochs},base_lr:{args.base_lr},batch_size:{args.batch_size},seed:{args.seed},loss:{args.loss}'

    logging.info(basic_info)
    logging.info(',epochs,train_loss,val_loss,x_ac,y_ac,z_ac,final_ac')

    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Grayscale(1)
        ])

    device = ('cuda' 
            if torch.cuda.is_available()
            else 'cpu')

    train_dataset = dataset.Speed_Dataset_3D(list_path_train,args.data_root,transform,1)
    train_loader = DataLoader(dataset = train_dataset,
                            batch_size = args.batch_size,
                            shuffle = True,
                            num_workers = 0)
    
    eval_dataset = dataset.Speed_Dataset_3D(list_path_eval,args.data_root,transform,1)
    eval_loader = DataLoader(dataset = eval_dataset,
                             batch_size = args.batch_size,
                             shuffle = True,
                             num_workers = 0)

    if args.net == 'CR': # 卷积中间加入残差
        model = Conv_Net_Residual(1, args.num_classes)
    elif args.net == 'SR': # 完全残差
        model = Simple_Residual(1,args.num_classes)
    elif args.net == 'SR1': # 完全残差
        model = Simple_Residual_1(1,args.num_classes)
    elif args.net == 'CN': # 纯卷积
        model = Conv_Net(1,args.num_classes)
    elif args.net == 'AN': # 全连接
        model = Activation_Net(args.img_size*args.img_size, 64*64, 32*32, args.num_classes) 
    elif args.net == 'CRN':
        model = Conv_Net_Residual_New(1, args.num_classes)
    
    model = model.to(device)

    if args.Resume:
        check_point = ''
        model.load_state_dict(torch.load(check_point))

    #输出模型结构
    #logging.info(model)

    if args.loss == 'MSE':
        loss = nn.MSELoss().to(device)
    elif args.loss == 'SmoothL1':
        loss = nn.SmoothL1Loss().to(device)

    optimizer = optim.Adam(model.parameters(), lr = args.base_lr)
    # 尽量动态lr，后期学习率需要比较小
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch: 1/(epoch+1))
    # scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)

    iterator = tqdm(range(args.max_epochs), ncols=70)
    loss_min = 0.03
    total_ac_min = 0.92
    size = len(eval_dataset)

    for i_epoch in iterator:
        # train
        loss_train = data_trainer_single(model,train_loader,loss,optimizer,device)
        # eval
        loss_eval, x_corr_num, y_corr_num, z_corr_num, total_corr_num = data_eval_speed_3D(model,eval_loader,loss,device)

        # if ((loss_eval < loss_min) and (i_epoch > 40) ):
        #     loss_min = loss_eval
        #     torch.save(model.state_dict(), f'{model_save_path}/{TIMESTAMP}_{i_epoch}eps_{loss_min:0.3f}loss.pth')
        if ((total_ac_min*size < total_corr_num) and (i_epoch > 40) ):
            total_ac_min = total_corr_num/size
            torch.save(model.state_dict(), f'{model_save_path}/{TIMESTAMP}_{i_epoch}eps_{total_ac_min:0.3f}ac.pth')

        scheduler.step()

        writer.add_scalar('train_loss',loss_train,i_epoch+1)
        writer.add_scalar('eval_loss',loss_eval,i_epoch+1)

        writer.add_scalar('x_correct',x_corr_num/size,i_epoch+1)
        writer.add_scalar('y_correct',y_corr_num/size,i_epoch+1)
        writer.add_scalar('z_correct',z_corr_num/size,i_epoch+1)
        writer.add_scalar('accuracy',total_corr_num/size,i_epoch+1)
        # logging.info(f'\nepoch:{i_epoch:0.3f} train_loss:{loss_train:0.3f} eval_loss:{loss_eval:0.3f} correct:{correct} accuracy:{correct/size:0.3f} \n')
        logging.info(f',{i_epoch+1:d},{loss_train:0.4f},{loss_eval:0.4f},{x_corr_num/size:0.4f},{y_corr_num/size:0.4f},{z_corr_num/size:0.4f},{total_corr_num/size:0.4f}')
    writer.close()
    torch.save(model.state_dict(), f'{model_save_path}/{TIMESTAMP}_{args.max_epochs}eps_{loss_eval:0.3f}loss.pth')
