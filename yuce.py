import os
import  cv2
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import  tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,BatchNormalization,Dense,Flatten,Dropout,MaxPooling2D,Input,Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import  matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
base_model_2 = tf.keras.applications.VGG16(
    include_top=False, weights='imagenet', input_shape=(112,112,3))
base_learning_rate = 0.0001
kernel_in3 = tf.constant([[
    [[-3.0, 0, 3.0], [-10.0, 0, 10.0], [-3.0, 0, 3.0]],
    [[-3.0, -10.0, -3.0], [0, 0, 0], [3.0, 1.0, 3.0]],
    [[-10.0, -3.0, 0], [-3.0, 0, 3.0], [0, 3.0, 10.0]],
    [[10.0, 3.0, 0], [3.0, 0, -3.0], [0, -3.0, -10.0]]]], shape=[4, 3, 3, 1], dtype=tf.float32)
print(kernel_in3.shape)
#
kernel_three = tf.constant(kernel_in3, dtype=tf.float32)
# kernel_five = tf.constant(x_in, dtype=tf.float32)
def VGG_16(input_shape=(224, 224, 3)):
    input_ = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    net = tf.nn.conv2d(input_, kernel_three, strides=[1, 1, 1, 1], padding="SAME")
    net = Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = MaxPooling2D()(net)

    # net1 = tf.nn.conv2d(input_, kernel_five, strides=[1, 1, 1, 1], padding="SAME")
    # net1 = Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu')(net1)
    # net1 = Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu')(net1)
    # net1 = MaxPooling2D()(net1)
    x = Concatenate()([x, net])
    # 第一次
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    net = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = MaxPooling2D()(net)

    # net1 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(net1)
    # net1 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(net1)
    # net1 = MaxPooling2D()(net1)
    x = Concatenate()([x, net])

    # 第二次

    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    net = Conv2D(64, kernel_size=1, strides=1, padding='same', activation='relu')(net)
    net = Conv2D(32, kernel_size=2, strides=1, padding='same', activation='relu')(net)
    net = Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu')(net)
    net = MaxPooling2D()(net)

    # net1 = Conv2D(32, kernel_size=1, strides=1, padding='same', activation='relu')(net1)
    # net1 = Conv2D(32, kernel_size=2, strides=1, padding='same', activation='relu')(net1)
    # net1 = Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu')(net1)
    # net1 = MaxPooling2D()(net1)
    x = Concatenate()([x, net])

    # 第三次
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    net = Conv2D(128, kernel_size=1, strides=1, padding='same', activation='relu')(net)
    net = Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu')(net)
    net = Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu')(net)
    net = MaxPooling2D()(net)

    # net1 = Conv2D(128, kernel_size=1, strides=1, padding='same', activation='relu')(net1)
    # net1 = Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu')(net1)
    # net1 = Conv2D(256, kernel_size=2, strides=1, padding='same', activation='relu')(net1)
    # net1 = MaxPooling2D()(net1)
    x = Concatenate()([x, net])

    # 第四次
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    # x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    # x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output_layer = tf.keras.layers.Dense(38, activation='softmax')(x)
    return Model(inputs=input_, outputs=output_layer)


model = VGG_16((224, 224, 3))
model.summary()
model.load_weights('D:/python/sheep_face/edge_Vgg_dir/train/events.out.tfevents.1653103437.DESKTOP-RVOOTND.31220.715.v2')
# model=densenet((224, 224, 3))
data='D:/python/sheep_face/lian_data/test1/sheep46/060 103_1.jpg'
# 换成图片路径
data=image.load_img(data)
x=image.img_to_array(data)
x=cv2.resize(x,(224,224))
x=x/255#要对图片进行预处理，其操作应与原来的训练集的操作一致
x=np.expand_dims(x,axis=0)
predict = model.predict(x)
def file_name(file_dir):
    listName = []
    #for root, dirs, files in os.walk(file_dir):
    for dir in os.listdir(file_dir):
        # print(dir)  # 当前目录路径
        listName.append(dir);
        #print(dirs)  # 当前路径下所有子目录
        # print(files)  # 当前路径下所有非目录子文件
    return  listName


file_dir = 'D:/python/sheep_face/lian_data/test1'     # 待取名称文件夹的绝对路径
listname = file_name(file_dir)

# 换成文件训练文件夹名称  名称要对应好
# def find(list,a):
#     for i in range(0,len(list)):
#         if list[i]==a:
#          return i
#     else:
#         return None
predict=np.argmax(predict,axis=1)
print('this image is:', listname[predict[0]])
# print(isinstance(predict,(str,int,list)))
plt.imshow(data)
plt.show()
