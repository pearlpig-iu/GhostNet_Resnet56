import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
# import Ghost_ResNet_conv1 as Ghost_ResNet
import Ghost_ResNet
import numpy as np
from PIL import Image
from torch.optim import lr_scheduler
import ctypes
import signal
import sys
import time
from model_trainer import ModelTrainer
from datetime import datetime
import matplotlib
import model_trainer

#player = ctypes.windll.kernel32
plt.rcParams['figure.dpi']=100
max_epoch = 100
batch_size = 4
lr = 0.001
max_iter = 5


# def sigint_handler(signum, frame):
#     global is_sigint_up
#     is_sigint_up = True
#     torch.save(model.state_dict(), 'gRes56_restore.weights')
#     print('Catched interrupt signal!')
#     # player.Beep(1000,1000)
#     sys.exit(0)
    
# signal.signal(signal.SIGINT, sigint_handler)
# signal.signal(signal.SIGTERM, sigint_handler)
# is_sigint_up = False



# class Cutout(object):
#     def __init__(self, hole_size):
#         self.hole_size = hole_size
#
#     def __call__(self, img):
#         return cutout(img, self.hole_size)
#
#
# def cutout(img, hole_size):
#     y = np.random.randint(32)
#     x = np.random.randint(32)
#
#     half_size = hole_size // 2
#
#     x1 = np.clip(x - half_size, 0, 32)
#     x2 = np.clip(x + half_size, 0, 32)
#     y1 = np.clip(y - half_size, 0, 32)
#     y2 = np.clip(y + half_size, 0, 32)
#
#     imgnp = np.array(img)
#
#     imgnp[y1:y2, x1:x2] = 0
#     img = Image.fromarray(imgnp.astype('uint8')).convert('RGB')
#     return img

train_dir = 'data/train'
valid_dir = 'data/test'
transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
validset = torchvision.datasets.ImageFolder(valid_dir, transform=transform)
validloader = torch.utils.data.DataLoader(dataset=validset, batch_size=batch_size)

classes = ('bingpian', 'bubingpian')
class_num = len(classes)
interval = 10
log_dir = 'result'
model = Ghost_ResNet.resnet56()

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[80,100], gamma=0.1)
optimizer = optim.Adam(model.parameters(), lr=lr)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
model.to(device)
# total_loss = []
# epoch_loss = []

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

def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()
# def show_loss(plt_loss):
#     plt.plot(range(len(plt_loss)), plt_loss)
#     plt.ylim((0, max(plt_loss)))
#     plt.show()

def get_lr():
    return optimizer.param_groups[0]['lr']

if __name__ == "__main__":
    start = 0
    iternum = 0
    #running_loss=0.
    # if resume == True:
    #     model.load_state_dict(torch.load("gRes56_restore.weights"))
    #     print('Resumed: weights reloaded')
    begin = time.time()

    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0

    for epoch in range(start, max_epoch):
        loss_train, acc_train, mat_train = ModelTrainer.train(trainloader, model, criterion, optimizer, epoch, max_epoch, device, class_num)
        loss_valid, acc_valid, mat_valid = ModelTrainer.valid(validloader, model, criterion, device, class_num)
        print(
            "Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}".format(
                epoch + 1, max_epoch, acc_train, acc_valid, loss_train, loss_valid,
                optimizer.param_groups[0]["lr"]))

        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)

        show_confMat(mat_train, classes, "train", log_dir, verbose=epoch == max_epoch - 1)
        show_confMat(mat_valid, classes, "valid", log_dir, verbose=epoch == max_epoch - 1)

        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)

        if epoch > (max_epoch / 2) and best_acc < acc_valid:
            best_acc = acc_valid
            best_epoch = epoch

            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_acc": best_acc}

            path_checkpoint = os.path.join(log_dir, "checkpoint_best.pkl")
            torch.save(checkpoint, path_checkpoint)
    print(" done ~~~~ {}, best acc: {} in :{}".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                                                      best_acc, best_epoch))


    #
    #
    #         ###save check point
    #         if (iternum+1) % 20 == 0:
    #             show_loss(epoch_loss)
    #             torch.save(net.state_dict(), 'gRes56_'+str(i)+'.weights')
    #         if iternum >= max_iter:
    #             break
    #         iternum += 1
    #     show_loss(epoch_loss)
    #     show_loss(total_loss)
    #     epoch_loss.clear()
    #     # if iternum >= max_iter:
    #     #     break
    # torch.save(net.state_dict(), 'gRes56.weights')
    # player.Beep(1000,1000)
    # end = time.time()
    # print("total time:",(end-begin)/3600)
    #
    #

