from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import  tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,BatchNormalization,Dense,Flatten,Dropout,MaxPooling2D,Input,Concatenate
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
    batch_size=5,
    class_mode='categorical',

)
validation_generator=validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224,224),
    batch_size=5,
    class_mode='categorical'

)
xtrain,ytrain=train_generator.next()
print(xtrain, ytrain)
kernel_in3 = tf.constant([[
    [[-3.0, 0, 3.0], [-10.0, 0, 10.0], [-3.0, 0, 3.0]],
    [[-3.0, -10.0, -3.0], [0, 0, 0], [3.0, 1.0, 3.0]],
    [[-10.0, -3.0, 0], [-3.0, 0, 3.0], [0, 3.0, 10.0]],
    [[10.0, 3.0, 0], [3.0, 0, -3.0], [0, -3.0, -10.0]]]], shape=[4, 3, 3, 1], dtype=tf.float32)
print(kernel_in3.shape)
#


kernel_in5 = tf.constant([[
    [[-1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -10.0, -3.0, -10.0, 0], [-1.0, -3.0, 0, 3.0, 1.0],
     [0, 10.0, 3.0, 10.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
    [[-1.0, -1.0, -1.0, 0, 1.0], [-1.0, -3.0, -10.0, 3.0, 1.0], [-1.0, -10.0, 0, 10.0, 1.0],
     [-1.0, -3.0, 10.0, 3.0, 1.0], [-1.0, 0, 1.0, 1.0, 1.0]],
    [[-1.0, 0, 1.0, 1.0, 1.0], [-1.0, -10.0, 3.0, 10.0, 1.0], [-1.0, -3.0, 0, 3.0, 1.0], [-1.0, -10.0, -3.0, 10.0, 1.0],
     [-1.0, -1.0, -1.0, 0, 1.0]],
    [[1.0, 1.0, 1.0, 1.0, 1.0], [0, 3.0, 10.0, 3.0, 1.0], [-1.0, -10.0, 0, 10.0, 1.0], [-1.0, -3.0, -10.0, -3.0, 0],
     [-1.0, -1.0, -1.0, -1.0, -1.0]]]], shape=[4, 5, 5, 1], dtype=tf.float32)





kernel_three = tf.constant(kernel_in3, dtype=tf.float32)
kernel_five = tf.constant(kernel_in5, dtype=tf.float32)



def Alxe_net(input_shape=(224, 224, 3)):
    input_ = Input(shape=input_shape)

    x = Conv2D(96, kernel_size=(11, 11), strides=(4, 4), padding='same', activation='relu')(input_)



    net = tf.nn.conv2d(input_, kernel_three, strides=[1, 1, 1, 1], padding="SAME")
    net = Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = MaxPooling2D()(net)
    net = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = MaxPooling2D()(net)
    x = Concatenate()([x, net])
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # net1 = tf.nn.conv2d(input_, kernel_five, strides=[1, 1, 1, 1], padding="SAME")
    # net1 = Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu')(net1)
    # net1 = Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu')(net1)
    # net1 = MaxPooling2D()(net1)


    x = Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x)

    net = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(net)
    net = MaxPooling2D()(net)
    x = Concatenate()([x, net])
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)



    # net1 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(net1)
    # net1 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(net1)
    # net1 = MaxPooling2D()(net1)


    x = Conv2D(348, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(348, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    net = Conv2D(64, kernel_size=1, strides=1, padding='same', activation='relu')(net)
    net = Conv2D(32, kernel_size=2, strides=1, padding='same', activation='relu')(net)
    net = Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu')(net)
    net = MaxPooling2D()(net)

    # net1 = Conv2D(32, kernel_size=1, strides=1, padding='same', activation='relu')(net1)
    # net1 = Conv2D(32, kernel_size=2, strides=1, padding='same', activation='relu')(net1)
    # net1 = Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu')(net1)
    # net1 = MaxPooling2D()(net1)
    x = Concatenate()([x, net])
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)



    x = Dense(4096, activation='relu')(x)


    x = Dense(1000, activation='relu')(x)

    output_layer = Dense(38, activation='softmax')(x)


    return Model(inputs=input_, outputs=output_layer)


model = Alxe_net((224, 224, 3))
model.summary()
model.compile(optimizer=optimizers.Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
tensorboard=tf.keras.callbacks.TensorBoard(log_dir='edge_alex_dir',
                                           histogram_freq=1,
                                           embeddings_freq=1)

chek=ModelCheckpoint(filepath='edge_face_alxe_weight/weights_{epoch:03d}-{loss:.4}-{val_accuracy:.5}.h5',
                     monitor='val_loss',
                     verbose=0,
                     )
#save the best model with lower validation loss

history = model.fit(train_generator,
                    steps_per_epoch=train_generator.n // 5,
                    epochs=200,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.n // 5,
                    callbacks=[chek, tensorboard],
                    shuffle=True
                    )
