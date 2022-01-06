# -*- coding: utf-8 -*-
"""
# @file name  : inference_in_test.py
# @author     : Sunweijie
# @date       : 2021-12-30
# @brief      : 测试test数据集上指标
"""
import matplotlib
matplotlib.use('agg')
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

# sys.path.append(os.path.abspath("../../"))
# sys.path.append("/home/tingsongyu/ghost_net_pytorch")
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model_trainer import ModelTrainer
from matplotlib import pyplot as plt
#from models.densenet import DenseNet121
#from models.lenet import LeNet

import Ghost_ResNet
import torchvision
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def show_confMat(confusion_mat, classes, set_name, out_dir, verbose=False):
    """
    混淆矩阵绘制
    :param confusion_mat:
    :param classes: 类别名
    :param set_name: trian/valid
    :param out_dir:
    :return:
    """
    cls_num = len(classes)
    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 显示

    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix' + set_name + '.png'))
    # plt.show()
    plt.close()

    if verbose:
        for i in range(cls_num):
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i]))))

if __name__ == '__main__':

    test_dir = '/home/Ghost_ResNet56-master/Ghost_ResNet56-master/data/test/'
    path_checkpoint = 'result/checkpoint_best.pkl'  # resnet-image
    #path_checkpoint = os.path.join(BASE_DIR, "..", "results/1/checkpoint_best.pth")
    #path_checkpoint = os.path.join(BASE_DIR, "..", "..", "results/03-06_16-54/checkpoint_best.pkl")  # vgg-cifar
    # valid_data = CifarDataset(data_dir=test_dir, transform=cfg.transforms_valid)
    # valid_loader = DataLoader(dataset=valid_data, batch_size=cfg.valid_bs, num_workers=cfg.workers)
    classes = ('bingpian', 'bubingpian')
    class_num =2 

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_data = torchvision.datasets.ImageFolder(
        test_dir,
        transform=transform
    )

    # 构建DataLoder
    test_loader = DataLoader(dataset=test_data, batch_size=4, num_workers=0, shuffle=False)



    log_dir = "../results"

    model = Ghost_ResNet.resnet56()

    check_p = torch.load(path_checkpoint, map_location="cpu", encoding='iso-8859-1')
    pretrain_dict = check_p["model_state_dict"]
    # print("best acc: {} in epoch:{}".format(check_p["best_acc"], check_p["epoch"]))
    #state_dict_cpu = state_dict_to_cpu(pretrain_dict)
    model.load_state_dict(pretrain_dict, strict=False)
    model.to(device)

    loss_f = nn.CrossEntropyLoss()

    loss_test, acc_test, mat_test = ModelTrainer.valid(test_loader, model, loss_f, device, class_num)

    #show_confMat(mat_test, classes, "test", log_dir, verbose=True)

    print("dataset: {}, acc: {} loss: {}".format(test_dir, acc_test, loss_test))

