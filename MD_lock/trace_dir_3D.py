import argparse
import torch
from torchvision import transforms
from net.net import *
from Det_single_label.utils import *
from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str,
                    default='Regression', help='Classify or Regression')
parser.add_argument('--net',type=str,
                    default='SR',
                    help='CR=>Conv_Net_Residual, SR=>Simple_Residual, AC=>Activation_Net, CR=Conv_Net')
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--data_root', type=str,
                    default='./data/8dir', help='input patch size of network input')
args = parser.parse_args()

if __name__ == '__main__':
    
    load_model_pth = f'./save/{args.mode}/{args.net}'+'/04-08-16-31_66eps_0.975ac.pth'

    trace_num = 11

    trace_true = f'./trace3D/{trace_num}/{trace_num}_speed.txt' #输入图片的列表
    speed = f'./trace3D/{trace_num}/{trace_num}_vs{trace_num}.txt' # 保存实际速度与预测速度
    trace = f'./trace3D/{trace_num}/{trace_num}_tr{trace_num}.txt' # 保存预测的轨迹与实际轨迹

    save_filepath = f'./trace3D/trace_SR_3407'

    device = ('cuda' 
        if torch.cuda.is_available()
        else 'cpu')

    if args.mode == 'Classify':
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Grayscale(1) 
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    if args.net == 'CR': # 卷积中间加入残差
        model = Conv_Net_Residual(1, args.num_classes)
    elif args.net == 'SR': # 完全残差
        model = Simple_Residual(1,args.num_classes)
    elif args.net == 'CN': # 纯卷积
        model = Conv_Net(1,args.num_classes)
    elif args.net == 'AN': # 全连接
        model = Activation_Net(args.img_size*args.img_size, 64*64, 32*32, args.num_classes) 
    elif args.net == 'CRN':
        model = Conv_Net_Residual_New(1, args.num_classes)

    model = model.to(device)

    model.load_state_dict(torch.load(load_model_pth))

    fp_sp = open(speed,'w')
    fp_tr = open(trace,'w')

    start_true = [0,0,0] # 初始位置
    start_pre = [0,0,0]
    frame = 5
    if args.mode == 'Classify':
        'ToDo:Draw_Trace'
    elif args.mode == 'Regression':
        'ToDo:Draw_Trace'
        with open(trace_true, 'r') as f:
            for lines in f:# 读取文件的每一行
                line = lines.rstrip('\n').split(',') # 去除掉尾部的回车，并按照逗号转为列表list
                img = transform(Image.open(line[0])).unsqueeze(0).to(device) # 获取轨迹图片并转为tensor,并扩展为4维数据
                output = model(img) # 得到预测结果
                output = torch.round(output)[0]
                fp_sp.write(f'{line[1]},{line[2]},{line[3]},{output[0]},{output[1]},{output[2]}\n')
                start_true = [start_true[0]+int(line[1])*frame,start_true[1]+int(line[2])*frame,start_true[2]+int(line[3])*frame]
                start_pre = [start_pre[0]+output[0]*frame,start_pre[1]+output[1]*frame,start_pre[2]+output[2]*frame]
                fp_tr.write(f'{start_true[0]},{start_true[1]},{start_true[2]},{start_pre[0]},{start_pre[1]},{start_pre[2]}\n')
            fp_sp.close()
            fp_tr.close()
    
    Ground_x = [0]
    Ground_y = [0]
    Ground_z = [0]
    Pre_x = [0]
    Pre_y = [0]
    Pre_z = [0]

    with open(trace,'r') as f:
        for lines in f:
            line_str = lines.rstrip('\n').split(',')
            line = [float(speed) for speed in line_str]
            Ground_x.append(line[0])
            Ground_y.append(line[1])
            Ground_z.append(line[2])
            Pre_x.append(line[3])
            Pre_y.append(line[4])
            Pre_z.append(line[5])

    ax3D = plt.axes(projection = '3d')
    colors = [0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100]
    ax3D.scatter3D(0,0,0,c='r')  #绘制散点图
    ax3D.plot3D(Ground_x,Ground_y,Ground_z,'v-y',label= 'Ground_Truth')    #绘制空间曲线
    ax3D.plot3D(Pre_x,Pre_y,Pre_z,'^:b',label= 'Pre_Trace')
    plt.legend()
    plt.show()
    

