import cv2
import os
import torch
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

from torch.utils.data import DataLoader
#from models.resnet import resnet56
import Ghost_ResNet
#from tools.common_tools import *
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
test_dir = 'data/test/bubingpian'
path_checkpoint = 'result/checkpoint_best.pkl' 

# label_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
label_names = ['bingpian', 'bubingpian']

def state_dict_to_cpu(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
    return new_state_dict


total_time = 0
num = 0

for file in os.listdir(test_dir):

    img = cv2.imread(test_dir + '/' + file)
    #model = resnet56()
    #model = replace_conv(model, GhostModule, arc="resnet56")
    

    model = Ghost_ResNet.resnet56()
    check_p = torch.load(path_checkpoint)
    pretrain_dict = check_p["model_state_dict"]
    #state_dict_cpu = state_dict_to_cpu(pretrain_dict)
    #model.load_state_dict(state_dict_cpu, strict=False)
    model.load_state_dict(pretrain_dict, strict=False)
    model.to(device)
    loss_f = torch. nn.CrossEntropyLoss()


    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #image.resize((112, 112,3))
    # inputs = image / 255
    inputs = image
    # inputs = (inputs - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    # 后续属于 图像分类的 通用处理
    # pytorch的格式   （H,W,C）   => (C,H,W)
    inputs = inputs.transpose(2, 0, 1)
    # (C, H, W) => (1, C, H, W)
    inputs = inputs[np.newaxis, :, :, :]
    # numpyArray => tensor
    inputs = torch.from_numpy(inputs)
    # dtype float32
    inputs = inputs.type(torch.float32)
    inputs = inputs.to(device)

    time1 = time.time()

    outputs = model(inputs)
    # _, predicted = torch.max(outputs.data, 1)
    # outputs = torch.abs(outputs)
    outputs = torch.softmax(outputs, dim=1)
    score, label_id = torch.max(outputs, dim=1)
    time2 = time.time()
    score, label_id = score.item(), label_id.item()
    label_name = label_names[label_id]
    t = time2 - time1
    print("img:{}, label:{},score:{}".format(file, label_name, score))
    print('time:',t)
    total_time += t
    num += 1
avg_time = total_time / num
print("toatl_time:{},avg_time:{}".format(total_time, avg_time))
