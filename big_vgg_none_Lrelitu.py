import os
import tensorflow as tf
import PIL
from tensorflow import keras
from keras import regularizers
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import Concatenate, Lambda, Input, ZeroPadding2D, AveragePooling2D
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout



def VGG_16(input_shape=(224, 224, 3)):

    input_ = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)


    x = Conv2D(128, kernel_size=(3, 3),strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)


    x = Conv2D(256, kernel_size=(3, 3),strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)


    x = Conv2D(512,kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)


    x = Conv2D(512, strides=(1, 1), kernel_size=(3, 3),padding='same', activation='relu')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = Conv2D(512,kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    x=tf.keras.layers.GlobalAveragePooling2D()(x)

    output_layer=tf.keras.layers.Dense(17, activation='softmax')(x)
    return Model(inputs=input_, outputs=output_layer)

model = VGG_16((224, 224, 3))

model.load_weights('D:/python/sheep_face/big_vgg_none_L2/weights_165-0.0289-0.85764.h5')
model.summary()
test_datagen = ImageDataGenerator(rescale=1./255)
seg_test = './big_data_test/'
test_generator = test_datagen.flow_from_directory(
        seg_test,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
labels = {0: '01', 1: '10', 2: '11', 3: '12', 4: '13', 5: '14', 6: '15', 7: '16', 8: '17', 9: '02', 10: '03', 11: '04', 12: '05', 13: '06', 14: '07', 15: '08', 16 : '09'}
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import cv2

prediction = []
original = []
image = []
count = 0
for i in os.listdir(seg_test):
  for item in os.listdir(os.path.join(seg_test,i)):
    #code to open the image
    img= PIL.Image.open(os.path.join(seg_test,i,item))
    #resizing the image to (256,256)
    img = img.resize((224,224))
    #appending image to the image list
    image.append(img)
    #converting image to array
    img = np.asarray(img, dtype= np.float32)
    #normalizing the image
    img = img / 255
    #reshaping the image in to a 4D array
    img = img.reshape(-1,224,224,3)
    #making prediction of the model
    predict = model.predict(img)
    #getting the index corresponding to the highest value in the prediction
    predict = np.argmax(predict)
    #appending the predicted class to the list
    prediction.append(labels[predict])
    #appending original class to the list
    original.append(i)
score = accuracy_score(original,prediction)
print("Test Accuracy : {}".format(score))

#visualizing the results
import random
fig=plt.figure(figsize = (100,100))
for i in range(8):
    j = random.randint(0,len(image))
    fig.add_subplot(8,1,i+1)
    plt.xlabel("Prediction -" + prediction[j] +"   Original -" + original[j])
    plt.imshow(image[j])
fig.tight_layout()
plt.show()
print(classification_report(np.asarray(original), np.asarray(prediction)))
plt.figure(figsize=(20,20))
cm = confusion_matrix(np.asarray(original), np.asarray(prediction))
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Original')
ax.set_title('Confusion_matrix')
plt.figure(figsize=(20,20))
cm = confusion_matrix(np.asarray(original), np.asarray(prediction))
plt.show()


def grad_cam(img):
    # Covert the image to array of type float32
    img = np.asarray(img, dtype=np.float32)

    # Reshape the image from (256,256,3) to (1,256,256,3)
    img = img.reshape(-1, 224, 224, 3)
    img_scaled = img / 255

    # Name of the layers we added to the base_model, you can find this in the model summaty
    # Every-time you run the model, check the summary, as the name would change or to avoid it
    # you can add name to each layer
    classification_layers = ["conv2d_13"]

    # Last convolutional layer in the base mdel, this woun't change as name has been already assigned to it.
    final_conv = model.get_layer("conv2d_14")

    # Create a model with original model input as input and the last conv_layer as the output
    final_conv_model = keras.Model(model.inputs, final_conv.output)

    # Then,we create the input for classification layer, which is the output of last conv layer
    # In our case, output produced by the conv layer is of the shape (1,3,3,2048)
    # Since, the classification input needs the features as input, we ignore the batch dimension

    classification_input = keras.Input(shape=final_conv.output.shape[1:])
    print(final_conv.output.shape[1:])

    # We iterate through the classification layers, to get the final layer and then, append
    # the layer as the output layer to the classification model.
    temp = classification_input
    for layer in classification_layers:
        temp = model.get_layer(layer)(temp)
        print('temp',temp)
    classification_model = keras.Model(classification_input, temp)

    # We use gradient tape to monitor the 'final_conv_output' to retrive the gradients
    # corresponding to the predicted class
    with tf.GradientTape() as tape:
        # Pass the image through the base model and get the feature map
        final_conv_output = final_conv_model(img_scaled)

        # Assign gradient tape to monitor the conv_output
        tape.watch(final_conv_output)

        # Pass the feature map through the classification model and use argmax to get the
        # index of the predicted class and then use the index to get the value produced by final
        # layer for that class
        prediction = classification_model(final_conv_output)
        print('predition',prediction.shape)

        predicted_class = tf.argmax(prediction[0][0][0])

        predicted_class_value = prediction[:, :, :, predicted_class]

    # Get the gradient corresponding to the predicted class based on feature map.
    # which is of shape (1,3,3,2048)
    gradient = tape.gradient(predicted_class_value, final_conv_output)

    # Since we need the filter values (2048), we reduce the other dimensions,
    # hich would result in a shape of (2048,)
    gradient_channels = tf.reduce_mean(gradient, axis=(0, 1, 2))

    # We then convert the feature map produced by last conv layer(1,6,6,1536) to (6,6,1536)

    final_conv_output = final_conv_output.numpy()[0]

    gradient_channels = gradient_channels.numpy()

    # We multiply the filters in the feature map produced by final conv layer by the
    # filter values that are used to get the predicted class. By doing this we inrease the
    # value of areas that helped in making the prediction and lower the vlaue of areas, that
    # did not contribute towards the final prediction
    for i in range(gradient_channels.shape[-1]):
        final_conv_output[:, :, i] *= gradient_channels[i]

    # We take the mean accross the channels to get the feature map
    heatmap = np.mean(final_conv_output, axis=-1)

    # Normalizing the heat map between 0 and 1, to visualize it
    heatmap_normalized = np.maximum(heatmap, 0) / np.max(heatmap)

    # Rescaling and converting the type to int
    heatmap = np.uint8(223 * heatmap_normalized)

    # Create the colormap
    color_map = plt.cm.get_cmap('jet')

    # get only the rb features from the heatmap
    color_map = color_map(np.arange(224))[:, :3]
    heatmap = color_map[heatmap]

    # convert the array to image, resize the image and then convert to array
    heatmap = keras.preprocessing.image.array_to_img(heatmap)
    heatmap = heatmap.resize((224, 224))
    heatmap = np.asarray(heatmap, dtype=np.float32)

    # Add the heatmap on top of the original image
    final_img = heatmap * 0.8 + img[0]
    final_img = keras.preprocessing.image.array_to_img(final_img)

    return final_img, heatmap_normalized

#Visualize the images in the dataset
import random
fig, axs = plt.subplots(6,3, figsize=(16,32))
count = 0
for _ in range(6):
  i = random.randint(0,len(image))
  gradcam, heatmap = grad_cam(image[i])
  axs[count][0].title.set_text("Original -" + original[i])
  axs[count][0].imshow(image[i])
  axs[count][1].title.set_text("Heatmap")
  axs[count][1].imshow(heatmap)
  axs[count][2].title.set_text("Prediction -" + prediction[i])
  axs[count][2].imshow(gradcam)
  count += 1

fig.tight_layout()
plt.show()