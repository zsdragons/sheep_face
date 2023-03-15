from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import  tensorflow as tf
from keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,BatchNormalization,Dense,Flatten,Dropout,MaxPooling2D,Input,Concatenate,Activation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import  matplotlib.pyplot as plt





def Bottleneck_2(input_ten, kernel_size, filters):
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, (1, 1))(input_ten)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1))(x)
    x = BatchNormalization()(x)

    x = tf.keras.layers.add([x, input_ten])
    x = Activation('relu')(x)
    return x


def Bottleneck_1(input_ten, kernel_size, filters, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    x = Conv2D(filters1, (1, 1), strides=strides)(input_ten)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides)(input_ten)
    shortcut = BatchNormalization()(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def Attention (input_ten):
    Attention_map = Bottleneck_1(26, 26, 52)(input_ten)
    Attention_map = Bottleneck_2(26, 26, 52)(Attention_map)
    Attention_map = Bottleneck_1(26, 26, 52)(Attention_map)
    Attention_map = Activation('sigmoid')(Attention_map)
    return Attention_map


def ARM_module(input1, input2):
    x = Concatenate()([input1, input2])

    att1 = Attention(x)
    res1 = Conv2D(1, kernel_size=1, strides=1, activation='relu')(x)
    res1 = tf.keras.layers.add([res1, input2])

    out = tf.matmul([res1, att1])
    out = tf.keras.layers.add([res1, out])

    out = Bottleneck_1(26, 26, 52)(out)
    return out


