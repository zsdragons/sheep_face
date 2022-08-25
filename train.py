import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import torchvision.datasets as dsets
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import pandas as pd
import seaborn as sns
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import ImageFolder
import torch
from Resnet_18 import resnet
from model import A2_Inception


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 加载数据
    # data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #                                                  torchvision.transforms.Normalize((0.401, 0.394, 0.388),
    #                                                                                   (0.142, 0.151, 0.157))])
    data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize((0.22710358, 0.2520602, 0.16506627),
                                                                                      (0.08900975, 0.08798268, 0.08762978))])
    # train_data = ImageFolder(r'D:\py-code\sheep_face\SheepBase\small_data\train', transform=data_transform)
    # test_data = ImageFolder(r'D:\py-code\sheep_face\SheepBase\small_data\test', transform=data_transform)
    train_data = ImageFolder(r'D:\py-code\bingchonghai\s62zm6djd2-1\Tomato_pest_image_enhancement\train',
                             transform=data_transform)   # 导入数据集
    test_data = ImageFolder(r'D:\py-code\bingchonghai\s62zm6djd2-1\Tomato_pest_image_enhancement\test',
                            transform=data_transform)    # 导入数据集
    train = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)

    # 加载模型
    # net = resnet(num_class=81)
    # net = A2_Inception(num_class=80)
    net = resnet(num_class=8)
    net = net.to(device)

    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()  # 交叉熵损失

    # 构建优化器
    LR = 0.001
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=LR) # 优化器

    epochs = 300
    best_epoch = 0
    best_acc = 0.0
    history = []

    save_path = r'D:\py-code\bingchonghai\model'

    train_data_size = len(train)
    test_data_size = len(test)

    for epoch in range(epochs):
        # train
        if epoch > 150:
            LR = 0.0001
            params = [p for p in net.parameters() if p.requires_grad]
            optimizer = optim.Adam(params, lr=LR)
        net.train()
        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0

        train_bar = tqdm(train)
        for step, data in enumerate(train_bar):
            img, labels = data
            optimizer.zero_grad()
            logits = net(img.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # 计算损失和准确率
            train_loss += loss.item()
            ret, prediction = torch.max(logits.data, dim=1)
            labels = labels.to(device)
            correct_count = prediction.eq(labels.data.view_as(prediction))
            acc = torch.mean(correct_count.type(torch.FloatTensor))
            train_acc += acc.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # test
        net.eval()
        acc = 0.0
        with torch.no_grad():
            test_bar = tqdm(test)
            for step, data in enumerate(test_bar):
                img, labels = data
                output = net(img.to(device))

                test_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
                loss = loss_function(output, labels.to(device))
                test_loss += loss.item()

                ret, predictions = torch.max(output.data, dim=1)
                labels = labels.to(device)
                correct_count = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_count.type(torch.FloatTensor))

                test_acc += acc.item()

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_test_loss = test_loss / test_data_size
        avg_test_acc = test_acc / test_data_size

        history.append([avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc])

        print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f  test_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, avg_train_loss, avg_train_acc, avg_test_loss, avg_test_acc))

        if avg_test_acc > best_acc:
            best_acc = avg_test_acc
            best_epoch = epoch +1

    save_path = save_path + r'/Resnet-18-' + str(best_acc) + 'pth'
    torch.save(net.state_dict(), save_path)

    dataset = 'bingchonghai'
    history = np.array(history)
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.savefig(dataset + '_loss_curve.png')
    plt.show()

    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(dataset + '_accuracy_curve.png')
    plt.show()

    print('Finish')

    return best_epoch, best_acc


if __name__ == '__main__':
    best_epoch, best_acc = main()
    print('best_epoch:%d, best_acc:%.4f' %(best_epoch, best_acc))