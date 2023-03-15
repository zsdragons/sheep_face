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




train_dir='D:/python/sheep_face/lian_data/train3'
validation_dir='D:/python/sheep_face/lian_data/test3'
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

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=15,
    class_mode='categorical',

)
validation_generator=validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224,224),
    batch_size=15,
    class_mode='categorical'

)



def VGG_16(input_shape=(224, 224, 3)):

    input_ = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    # x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    # x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output_layer=tf.keras.layers.Dense(38, activation='softmax')(x)
    return Model(inputs=input_, outputs=output_layer)

model = VGG_16((224, 224, 3))
model.summary()
model.compile(optimizer=optimizers.Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
tensorboard=tf.keras.callbacks.TensorBoard(log_dir='face_vgg_dir',
                                           histogram_freq=1,
                                           embeddings_freq=1)

chek=ModelCheckpoint(filepath='face_vgg_weight/weights_{epoch:03d}-{loss:.4}-{val_accuracy:.5}.h5',
                     monitor='val_loss',
                     verbose=0,
                     )


history = model.fit(train_generator,
                    steps_per_epoch=train_generator.n // 15,
                    epochs=200,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.n // 15,
                    callbacks=[chek, tensorboard],
                    shuffle=True
                    )
