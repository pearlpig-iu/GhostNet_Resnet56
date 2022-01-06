# -*- coding: utf-8 -*-
"""
# @file name  : model_trainer.py
# @author     : TingsongYu https://github.com/TingsongYu
# @date       : 2020-02-29
# @brief      : 模型训练类
"""
import torch
import time
import numpy as np
from functools import reduce


class ModelTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, epoch_id, max_epoch, device, class_num):
        model.train()

        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []
        log_interval = 10

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            optimizer.zero_grad()
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)

            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 统计loss
            loss_sigma.append(loss.item())
            acc_avg = conf_mat.trace() / conf_mat.sum()

            # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
            if i % log_interval == log_interval - 1:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch_id + 1, max_epoch, i + 1, len(data_loader), np.mean(loss_sigma), acc_avg))

        return np.mean(loss_sigma), acc_avg, conf_mat

    @staticmethod
    def valid(data_loader, model, loss_f, device, class_num):
        # model.eval()
        model.train()
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma = []

        for i, data in enumerate(data_loader):
            # print(data[0])
            # print('------------------------------------')
            # print(data[1])

            inputs, labels = data
            # print(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            # if labels == torch.Tensor([1]):
            outputs = model(inputs)
            loss = loss_f(outputs, labels)
            print(outputs)
            # print(labels)
            # 统计预测信息
            logit = torch.softmax(outputs, dim=1)
            # print(logit)
            _, predicted = torch.max(logit.data, 1)
            print(labels,predicted)
            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 统计loss
            loss_sigma.append(loss.item())

            acc_avg = conf_mat.trace() / conf_mat.sum()

        return np.mean(loss_sigma), acc_avg, conf_mat
