import torch
import logging
from torch import nn, optim
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
from datetime import datetime

'''
分类问题单一标签
'''
def data_trainer_single(model, train_loader, loss_fn, 
                        optimizer,device = 'cpu', writer=None):
    #设为训练状态
    model.train()
    loss_sum = 0
    # 训练模型
    for i_batch, data in enumerate(train_loader):
        img, target = data
        img, target = img.to(device), target.to(device)

        outputs = model(img)
        # outputs = torch.round(outputs)
        loss = loss_fn(outputs, target)
    
        optimizer.zero_grad()#梯度清零
        loss.backward()#反向传播
        optimizer.step()#梯度下降

        loss_sum += loss.item()

    return loss_sum / (i_batch+1)


def data_eval_single(model, eval_loader, loss_fn, 
                     device = 'cpu', writer=None):
    test_loss = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for i_batch, data in enumerate(eval_loader):
            img, target = data
            img, target = img.to(device), target.to(device)

            output = model(img)
            loss = loss_fn(output, target).item()

            test_loss += loss  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    return test_loss/(i_batch + 1), correct

def data_eval_speed(model, eval_loader, loss_fn, device = 'cpu', writer=None):
    test_loss = 0
    model.eval()
    x_corr_num = 0
    y_corr_num = 0
    total_corr_num = 0
    with torch.no_grad():
        for i_batch, data in enumerate(eval_loader):
            img, target = data
            img, target = img.to(device), target.to(device)

            output = model(img)
            loss = loss_fn(output, target).item()

            test_loss += loss  # sum up batch loss
            
            output = torch.round(output)
            out = output.permute(1,0)
            tar = target.permute(1,0)
            x_out, y_out = out
            x_tar, y_tar = tar
            x_equal = x_out.eq(x_tar)
            y_equal = y_out.eq(y_tar)
            x_corr_num =x_corr_num + x_equal.sum().item()
            y_corr_num =y_corr_num + y_equal.sum().item()

            for index,corr in enumerate(x_equal):
                if corr and y_equal[index]:
                    total_corr_num = total_corr_num + 1

    return test_loss/(i_batch + 1), x_corr_num, y_corr_num, total_corr_num

def data_eval_speed_3D(model, eval_loader, loss_fn, device = 'cpu', writer=None):
    test_loss = 0
    model.eval()
    x_corr_num = 0
    y_corr_num = 0
    z_corr_num = 0
    total_corr_num = 0
    with torch.no_grad():
        for i_batch, data in enumerate(eval_loader):
            img, target = data
            img, target = img.to(device), target.to(device)

            output = model(img)
            loss = loss_fn(output, target).item()

            test_loss += loss  # sum up batch loss
            
            output = torch.round(output)
            out = output.permute(1,0)
            tar = target.permute(1,0)
            x_out, y_out, z_out = out
            x_tar, y_tar, z_tar = tar
            x_equal = x_out.eq(x_tar)
            y_equal = y_out.eq(y_tar)
            z_equal = z_out.eq(z_tar)
            x_corr_num =x_corr_num + x_equal.sum().item()
            y_corr_num =y_corr_num + y_equal.sum().item()
            z_corr_num =z_corr_num + z_equal.sum().item()

            for index,corr in enumerate(x_equal):
                if corr and y_equal[index] and z_equal[index]:
                    total_corr_num = total_corr_num + 1

    return test_loss/(i_batch + 1), x_corr_num, y_corr_num, z_corr_num, total_corr_num


def get_accuracy(target, pre):
    x_target, y_target = target
    x_pre, y_pre = pre
    return























if __name__ == '__main__':
    #设置输出内容的格式，分别对应时间，日志等级，要输出信息
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    selfdef_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(selfdef_fmt)

    log_file = 'testfun.txt'
    # 设置一个输出到文件的Handler
    handler_test = logging.FileHandler(log_file,'w') 
    # 设置ERROR级别，只有高于或等于该级别的信息才会由Handler输出到对应位置
    handler_test.setLevel('ERROR')  
    # 设置输出信息的格式         
    handler_test.setFormatter(formatter)

    # 设置一个输出到stdout的Handler
    handler_control = logging.StreamHandler()    # stdout to console
    # 设置INFO级别，只有高于或等于该级别的信息才会由Handler输出到对应位置
    handler_control.setLevel('INFO')             # 设置INFO级别
    # 设置输出信息的格式
    handler_control.setFormatter(formatter)

    # 设置Logger所有信息将从Logger输出到Handler
    logger = logging.getLogger()
    # 必须要设置Logger的输出等级，否则无法输出信息
    logger.setLevel('DEBUG')           #设置了这个才会把debug及以上的输出到控制台

    #添加handler
    logger.addHandler(handler_test)    
    logger.addHandler(handler_control)

    epoch = 10
    loss = 0.023
    #打印中文会乱码
    #通过logger输出不同级别的信息，对应formatter中的%(message)s部分
    logger.debug('debug,打印一些数据的值,用于debug判断')
    logger.info('info,一般的信息输出')
    logger.info('epoch: %d, loss: %f' %(epoch, loss))
    logger.warning('waring,用来用来打印警告信息')
    logger.error('error,一般用来打印一些错误信息')
    logger.error('epoch: %d, loss: %f' %(epoch, loss))
    logger.critical('critical,用来打印一些致命的错误信息,等级最高')