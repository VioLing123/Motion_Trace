from torch import nn
import torch.functional as F
 
class simpleNet(nn.Module):
    """
    定义了一个简单的三层全连接神经网络，每一层都是线性的
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

"""
在上面的simpleNet的基础上，在每层的输出部分添加了激活函数，输入会直接平坦化，不需要提前转换
""" 
class Activation_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim, activate = 'Relu'):
        super(Activation_Net, self).__init__()
        if(activate == 'Relu'):
            self.FL = nn.Flatten()
            self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
            self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
            self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        elif(activate == 'LeakyRelu'):
            self.FL = nn.Flatten()
            self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.LeakyReLU(True))
            self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.LeakyReLU(True))
            self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        """
        这里的Sequential()函数的功能是将网络的层组合到一起。
        """
 
    def forward(self, x):
        x = self.FL(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
 
class Batch_Net(nn.Module):
    """
    在上面的Activation_Net的基础上，增加了一个加快收敛速度的方法——批标准化
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Activation_Net_Multi(nn.Module):
    """
    在上面的simpleNet的基础上，在每层的输出部分添加了激活函数
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net_Multi, self).__init__()
        self.class_filter = nn.ModuleList()
        for channel in out_dim:
            self.class_filter.append(Activation_Net(in_dim, n_hidden_1, n_hidden_2, channel))
        """
        这里的Sequential()函数的功能是将网络的层组合到一起。
        """
 
    def forward(self, x, label_classes):
        output = {}
        for index in range(len(label_classes)):
            output[label_classes[index]] = self.class_filter[index](x)
        return output


class Conv_Net(nn.Module):
    def __init__(self,in_channel, out_dim) -> None:
        super(Conv_Net, self).__init__()
        self.Conv_Net1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size= 3, stride= 2, padding= 1), # 64
            nn.ReLU(),
            nn.MaxPool2d(2), # 32
            nn.Conv2d(64, 128, kernel_size= 3, stride= 1, padding= 1),# 32 
            nn.ReLU(),
            nn.MaxPool2d(2),# 16
            nn.Conv2d(128, 256, kernel_size= 3, stride= 2, padding= 1), # 8 
            nn.ReLU(),
            nn.MaxPool2d(2) # 4
        )
        # self.Net_Con = nn.Flatten()
        self.FC = Activation_Net(256*4*4,512,128,out_dim)

    def forward(self, x):
        x = self.Conv_Net1(x)
        # x = self.Net_Con(x)
        x = self.FC(x)
        return x
    
class Block_Residual(nn.Module):
    def __init__(self, in_channel, out_channel, use_1x1=0, stride=1) -> None:
        super(Block_Residual,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel,out_channel, kernel_size=3, stride=stride,padding= 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel, kernel_size=3, padding= 1),
            nn.BatchNorm2d(out_channel)
        )
        if use_1x1:
            self.Residual = nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=stride)
        else:
            self.Residual = None
        self.Relu = nn.ReLU()

    def forward(self, x):
        y = self.block(x)
        if self.Residual:
            x = self.Residual(x)
        x += y
        x = self.Relu(x)
        return x


class Conv_Net_Residual_New(nn.Module):
    def __init__(self,in_channel, out_dim) -> None:#input 128
        super(Conv_Net_Residual_New, self).__init__()
        self.Conv_Net1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size= 3, stride= 2, padding= 1), #64
            nn.LeakyReLU(),
            nn.MaxPool2d(2), # 32
        )
        self.Residual1 = Block_Residual(64, 128, 1, 2) # 16
        self.Conv_Net2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size= 3, stride= 1, padding= 1), # 16
            nn.LeakyReLU(),
            nn.MaxPool2d(2), # 8
        )
        self.aver = nn.AvgPool2d(kernel_size=8,stride=1)
        # self.FL = nn.Flatten()
        self.FC = Activation_Net(256,128,64,out_dim,'LeakyRelu')
        # self.sig = nn.Sigmoid()
        # self.RL = nn.ReLU()

    def forward(self, x):
        x = self.Conv_Net1(x)
        x = self.Residual1(x)
        x = self.Conv_Net2(x)
        x = self.aver(x)
        x = self.FC(x)
        # x = self.sig(self.FC(x))
        return x
    
class Conv_Net_Residual(nn.Module):
    def __init__(self,in_channel, out_dim) -> None:#input 128
        super(Conv_Net_Residual, self).__init__()
        self.Conv_Net1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size= 3, stride= 2, padding= 1), #64
            nn.ReLU(),
            nn.MaxPool2d(2) # 32
        )
        self.Residual1 = Block_Residual(64, 128, 1, 2) # 16
        self.Conv_Net2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size= 3, stride= 1, padding= 1), # 16
            nn.ReLU(),
            nn.MaxPool2d(2) # 8
        )
        self.aver = nn.AvgPool2d(kernel_size=8,stride=1)
        # self.FL = nn.Flatten()
        self.FC = Activation_Net(256,128,64,out_dim)

    def forward(self, x):
        x = self.Conv_Net1(x)
        x = self.Residual1(x)
        x = self.Conv_Net2(x)
        x = self.aver(x)
        # x = self.FL(x)
        x = self.FC(x)
        return x


#output = (H+2Padding-kernel)/stride + 1 

class Simple_Residual(nn.Module):
    def __init__(self, in_channel, num_classes) -> None:
        super(Simple_Residual, self).__init__()
        #in_channel = 1, H = 1x128x128 -> 64x126x126
        self.Conv1 = nn.Conv2d(in_channel, 64, kernel_size=5, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(2) # -> 64x63x63 
        self.block1 = Block_Residual(64,128,1,2) # -> 128x32x32
        self.block2 = Block_Residual(128,256,1,2) # -> 256x16x16
        self.block3 = Block_Residual(256,512,1,2) # -> 512x8x8
        self.aver = nn.AvgPool2d(kernel_size=8,stride=1) # ->512x1x1
        self.FL = nn.Flatten()
        self.FC = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,num_classes),
            # nn.Sigmoid()
            )
        
    def forward(self, x):
        x = self.Conv1(x)
        x = self.maxpool1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.aver(x)
        x = self.FL(x)
        x = self.FC(x)
        return x