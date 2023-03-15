from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import  tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,BatchNormalization,Dense,Flatten,Dropout,MaxPooling2D,Input,Concatenate, UpSampling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import  matplotlib.pyplot as plt
if __name__ == '__main__':
    # GPU settings
    gpus= tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


train_dir='D:/python/sheep_face/lian_data/train1'
validation_dir='D:/python/sheep_face/lian_data/test1'
# validation_dir='‪D:/dataset/VAE/valid'
train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

validation_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
# test_datagent=ImageDataGenerator(rescale=1./255)
train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=1,
    class_mode='sparse',

)
validation_generator=validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224,224),
    batch_size=1,
    class_mode='sparse'

)


def create_model():
    ## unet网络结构下采样部分
    # 输入层 第一部分
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)  # 256*256*64
    # 下采样
    x1 = tf.keras.layers.MaxPooling2D(padding="same")(x)  # 128*128*64

    # 卷积 第二部分
    x1 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)  # 128*128*128
    # 下采样
    x2 = tf.keras.layers.MaxPooling2D(padding="same")(x1)  # 64*64*128

    # 卷积 第三部分
    x2 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)  # 64*64*256
    # 下采样
    x3 = tf.keras.layers.MaxPooling2D(padding="same")(x2)  # 32*32*256

    # 卷积 第四部分
    x3 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)  # 32*32*512
    # 下采样
    x4 = tf.keras.layers.MaxPooling2D(padding="same")(x3)  # 16*16*512
    # 卷积  第五部分
    x4 = tf.keras.layers.Conv2D(1024, 3, padding="same", activation="relu")(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.Conv2D(1024, 3, padding="same", activation="relu")(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)  # 16*16*1024

    ## unet网络结构上采样部分

    # 反卷积 第一部分      512个卷积核 卷积核大小2*2 跨度2 填充方式same 激活relu
    x5 = tf.keras.layers.Conv2DTranspose(512, 2, strides=2,
                                         padding="same",
                                         activation="relu")(x4)  # 32*32*512
    x5 = tf.keras.layers.BatchNormalization()(x5)
    x6 = tf.concat([x3, x5], axis=-1)  # 合并 32*32*1024
    # 卷积
    x6 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x6)
    x6 = tf.keras.layers.BatchNormalization()(x6)
    x6 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x6)
    x6 = tf.keras.layers.BatchNormalization()(x6)  # 32*32*512

    # 反卷积 第二部分
    x7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2,
                                         padding="same",
                                         activation="relu")(x6)  # 64*64*256
    x7 = tf.keras.layers.BatchNormalization()(x7)
    x8 = tf.concat([x2, x7], axis=-1)  # 合并 64*64*512
    # 卷积
    x8 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x8)
    x8 = tf.keras.layers.BatchNormalization()(x8)
    x8 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x8)
    x8 = tf.keras.layers.BatchNormalization()(x8)  # #64*64*256

    # 反卷积 第三部分
    x9 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2,
                                         padding="same",
                                         activation="relu")(x8)  # 128*128*128
    x9 = tf.keras.layers.BatchNormalization()(x9)
    x10 = tf.concat([x1, x9], axis=-1)  # 合并 128*128*256
    # 卷积
    x10 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x10)
    x10 = tf.keras.layers.BatchNormalization()(x10)
    x10 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x10)
    x10 = tf.keras.layers.BatchNormalization()(x10)  # 128*128*128

    # 反卷积 第四部分
    x11 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2,
                                          padding="same",
                                          activation="relu")(x10)  # 256*256*64
    x11 = tf.keras.layers.BatchNormalization()(x11)
    x12 = tf.concat([x, x11], axis=-1)  # 合并 256*256*128
    # 卷积
    x12 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x12)
    x12 = tf.keras.layers.BatchNormalization()(x12)
    x12 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x12)
    x12 = tf.keras.layers.BatchNormalization()(x12)  # 256*256*64

    # 输出层 第五部分
    output = tf.keras.layers.Conv2D(38, 1, padding="same")(x12)  # 256*256*34

    return tf.keras.Model(inputs=inputs, outputs=output)


model = create_model()



model.summary()
model.compile(optimizer=optimizers.Adam(lr=0.0001),
              loss='mse',
              metrics=['accuracy'])
#using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
tensorboard=tf.keras.callbacks.TensorBoard(log_dir='All_net_dir/unet_dir',
                                           histogram_freq=1,
                                           embeddings_freq=1)

chek=ModelCheckpoint(filepath='All_net_weight/unet_weight/weights_{epoch:03d}-{loss:.4}-{val_accuracy:.5}.h5',
                     monitor='val_loss',
                     verbose=0,
                     )
#save the best model with lower validation loss

history = model.fit(train_generator,
                    steps_per_epoch=train_generator.n // 1,
                    epochs=50,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.n // 1,
                    callbacks=[chek, tensorboard],
                    shuffle=True
                    )
