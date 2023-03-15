import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.font_manager import FontProperties
import csv

'''读取csv文件'''


def readcsv(files):
    csvfile = open(files,'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append(float(row[2]))
        x.append(float(row[1]))
    return x,y



mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
plt.figure()

# x1, y1 = readcsv("D:/python/sheep_face/biande/alex-gai-acc.csv")
# plt.plot(x1, y1, color='g', label='L-VGG19_loss')
# #
# x, y = readcsv("D:/python/sheep_face/biande/alex-run-validation-tag-epoch_accuracy (1).csv")
# plt.plot(x, y, 'b', label='edge_L-VGG19_loss')
# # # #
# # #
# 4
x4, y4 = readcsv("F:/张世龙硕士-2019-2021/硕士论文/各种图/csv/源网络/resnet/acc_t.csv")
plt.plot(x4, y4, color='lime', label='AE-ARM_ResNet_Train_acc')

x4, y4 = readcsv("F:/张世龙硕士-2019-2021/硕士论文/各种图/csv/源网络/resnet/acc_v.csv")
plt.plot(x4, y4, color='r', label='AE-ARM_ResNet_Val_acc')

x2, y2 = readcsv("F:/张世龙硕士-2019-2021/硕士论文/各种图/csv/resnet/res_v.csv")
plt.plot(x2, y2, 'black', label='ResNet_Train_acc')

x2, y2 = readcsv("F:/张世龙硕士-2019-2021/硕士论文/各种图/csv/resnet/rest.csv")
plt.plot(x2, y2, 'blue', label='ResNet_Val_acc')

# x2, y2 = readcsv("F:/Z_D/csv/5.csv")
# plt.plot(x2, y2, 'green', label='MRFEM-YOLOv4+K-maeans++')
# x4, y4 = readcsv("F:/Z_D/csv/6.csv")
# plt.plot(x4, y4, color='yellow', label='MRFEM-YOLOv4+K-maeans++(D)')
# x2, y2 = readcsv("F:/Z_D/csv/7.csv")
# plt.plot(x2, y2, 'green', label='MRFEM-YOLOv4+K-maeans++')
# x4, y4 = readcsv("F:/Z_D/csv/8.csv")
# plt.plot(x4, y4, color='yellow', label='MRFEM-YOLOv4+K-maeans++(D)')


# # 2


# # 1
# x4, y4 = readcsv("E:/硕士论文/loss/1/run-loss_2022_04_14_09_21_43_train-tag-epoch_loss.csv")
# plt.plot(x4, y4, color='g', label='Triplet_loss')
#
# x2, y2 = readcsv("E:/硕士论文/loss/1/run-loss_2022_04_14_09_21_43_validation-tag-epoch_loss.csv")
# plt.plot(x2, y2, 'r', label='Loss')

plt.ylim(0,1)
plt.xlim(0, 200)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.legend(fontsize=14)
plt.show()