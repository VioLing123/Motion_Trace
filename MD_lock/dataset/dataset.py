from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch import tensor
import os
from PIL import Image
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库
import numpy as np

'''
文件结构：
test
|---dir1 #不同种类的图片
|   |-----|- 1.jpg
|         |- 2.jpg
|         |- ...
|---dir2
|   |-----|- 1.jpg
|         |- 2.jpg
|         |- ...
|---test.csv #不同种类的图片位置及其label
'''

'''
重写的Dataset
'''
class Move_Dataset(Dataset):
    def __init__(self,names_file,transform=None,labels_classes=[]):
        self.names_file = names_file #.txt文件路径
        self.transform = transform #数据预处理
        self.size = 0 #数据集数量
        self.names_list = [] #数据集路径列表
        self.labels_classes = labels_classes
        
        if len(labels_classes) <= 0:
            print('label\'s classes must be more than one !')

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        with open(self.names_file) as f:
            for lines in f: #循环读取.txt文件总每行数据信息
                self.names_list.append(lines.rstrip('\n'))
                self.size += 1
        
    def __len__(self):
        return self.size
    
    def __getitem__(self,index):
        image_path = self.names_list[index].split(' ')[0] #获取图片数据路径
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        
        #读取对应的标签
        labels = {}
        for label in range(len(self.labels_classes)):
            labels[self.labels_classes[label]] = tensor(int(self.names_list[index].split(' ')[label+1]))

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return image,labels


class Move_Dataset_Single(Dataset):
    '''
    data_root_path : 数据跟目录,目录下为train、val两个文件夹
    list_path : 对应txt文件的位置
    '''
    def __init__(self,list_path,data_root_path,transform=None):
        self.data_root_path = data_root_path
        self.list_path = list_path #.txt文件路径
        self.transform = transform #数据预处理
        self.size = 0 #数据集数量
        self.names_list = [] #数据集路径列表

        if not os.path.isfile(self.list_path):
            print(self.list_path + ' does not exist!')
        with open(self.list_path) as f:
            for lines in f: #循环读取.txt文件总每行数据信息
                self.names_list.append(lines.rstrip('\n'))
                self.size += 1
        
    def __len__(self):
        return self.size
    
    def __getitem__(self,index):
        image_path = self.data_root_path + self.names_list[index].split(' ')[0] #获取图片数据路径
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        
        #读取对应的标签
        label = tensor(int(self.names_list[index].split(' ')[1]))

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return image,label

class Speed_Dataset(Dataset):
    '''
    data_root_path : 数据跟目录,目录下为train、val两个文件夹
    list_path : 对应txt文件的位置
    normalize : 归一化,所有速度值除以该值
    '''
    def __init__(self,list_path,data_root_path,transform=None, normalize=1):
        self.data_root_path = data_root_path
        self.list_path = list_path #.txt文件路径
        self.transform = transform #数据预处理
        self.size = 0 #数据集数量
        self.names_list = [] #数据集路径列表
        self.normalize = normalize

        if not os.path.isfile(self.list_path):
            print(self.list_path + ' does not exist!')
        with open(self.list_path) as f:
            for lines in f: #循环读取.txt文件总每行数据信息
                self.names_list.append(lines.rstrip('\n'))
                self.size += 1
        
    def __len__(self):
        return self.size
    
    def __getitem__(self,index):
        image_path = self.data_root_path + self.names_list[index].split(',')[0] #获取图片数据路径
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        
        #读取对应的标签
        label = tensor([int(self.names_list[index].split(',')[1])/self.normalize,int(self.names_list[index].split(',')[2])/self.normalize],
                       dtype=torch.float)

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return image,label

class Speed_Dataset_3D(Dataset):
    '''
    data_root_path : 数据跟目录,目录下为train、val两个文件夹
    list_path : 对应txt文件的位置
    normalize : 归一化,所有速度值除以该值
    '''
    def __init__(self,list_path,data_root_path,transform=None, normalize=1):
        self.data_root_path = data_root_path
        self.list_path = list_path #.txt文件路径
        self.transform = transform #数据预处理
        self.size = 0 #数据集数量
        self.names_list = [] #数据集路径列表
        self.normalize = normalize

        if not os.path.isfile(self.list_path):
            print(self.list_path + ' does not exist!')
        with open(self.list_path) as f:
            for lines in f: #循环读取.txt文件总每行数据信息
                self.names_list.append(lines.rstrip('\n'))
                self.size += 1
        
    def __len__(self):
        return self.size
    
    def __getitem__(self,index):
        image_path = self.data_root_path + self.names_list[index].split(',')[0] #获取图片数据路径
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        
        #读取对应的标签
        label = tensor([int(self.names_list[index].split(',')[1])/self.normalize,int(self.names_list[index].split(',')[2])/self.normalize,int(self.names_list[index].split(',')[3])/self.normalize],
                       dtype=torch.float)

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        return image,label



def draw_confusion_martix(file, labels_name):
    true_label = []
    pre_label = []
    with open(file, 'r') as f:
        for lines in f:
            data = lines.rstrip('\n').split(',')
            true_label.append(int(data[0]))
            pre_label.append(int(data[1]))
    cm = confusion_matrix(true_label,pre_label)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, cmap=plt.get_cmap('Blues'))    # 在特定的窗口上显示图像
    plt.title('HAR Confusion Matrix')    # 图像标题

    for first_index in range(len(cm)):  # 第几行
        for second_index in range(len(cm[first_index])):  # 第几列
            temp = cm[first_index][second_index]
            if temp == 0.0 or temp == 100.0:
                plt.text(second_index, first_index, int(temp), va='center',
                        ha='center' )
            else:
                plt.text(second_index, first_index, r'{0:.2f}'.format(temp), va='center',
                        ha='center' )
                
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')
    plt.show()
    return

# def Catch_Trace():
#     return img, speed

