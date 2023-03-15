import  cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.keras.models import Model
config = ConfigProto()
config.gpu_options.allow_growth = True
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
model=load_model('D:/python/sheep_face/py_weight/weights_099-0.03971-0.95796.h5',compile=None)
model.summary()
layer_name='max_pooling2d_8'
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)

data='D:/python/sheep_face/img/py4.jpg'
data=image.load_img(data)
x=image.img_to_array(data)
x=cv2.resize(x,(224,224))
x=x/255#要对图片进行预处理，其操作应与原来的训练集的操作一致
x=np.expand_dims(x,axis=0)
print(x.shape)
x=intermediate_layer_model(x)
print(x.shape)
# plt.matshow(x[0,:,:,3],cmap='viridis')
# plt.matshow(x[0,:,:,4],cmap='viridis')
# plt.matshow(x[0,:,:,5],cmap='viridis')
# plt.matshow(x[0,:,:,6],cmap='viridis')
# plt.matshow(x[0,:,:,7],cmap='viridis')
# plt.matshow(x[0,:,:,8],cmap='viridis')
# plt.matshow(x[0,:,:,9],cmap='viridis')
# plt.matshow(x[0,:,:,10],cmap='viridis')
# plt.matshow(x[0,:,:,11],cmap='viridis')
# plt.matshow(x[0,:,:,12],cmap='viridis')
# plt.matshow(x[0,:,:,13],cmap='viridis')
# plt.matshow(x[0,:,:,14],cmap='viridis')
# plt.matshow(x[0,:,:,15],cmap='viridis')
# plt.matshow(x[0,:,:,16],cmap='viridis')
# plt.matshow(x[0,:,:,17],cmap='viridis')
# plt.matshow(x[0,:,:,18],cmap='viridis')
# plt.matshow(x[0,:,:,19],cmap='viridis')
# plt.matshow(x[0,:,:,20],cmap='viridis')
# plt.matshow(x[0,:,:,21],cmap='viridis')
# plt.matshow(x[0,:,:,22],cmap='viridis')
# plt.matshow(x[0,:,:,23],cmap='viridis')
# plt.matshow(x[0,:,:,22],cmap='viridis')
# plt.matshow(x[0,:,:,23],cmap='viridis')
# plt.matshow(x[0,:,:,24],cmap='viridis')
# plt.matshow(x[0,:,:,25],cmap='viridis')
# plt.matshow(x[0,:,:,26],cmap='viridis')
# plt.matshow(x[0,:,:,27],cmap='viridis')
# plt.matshow(x[0,:,:,28],cmap='viridis')
# plt.matshow(x[0,:,:,29],cmap='viridis')



plt.matshow(x[0,:,:,30],cmap='viridis')
plt.matshow(x[0,:,:,40],cmap='viridis')
plt.matshow(x[0,:,:,50],cmap='viridis')
plt.matshow(x[0,:,:,60],cmap='viridis')
plt.matshow(x[0,:,:,70],cmap='viridis')
plt.matshow(x[0,:,:,80],cmap='viridis')
plt.matshow(x[0,:,:,90],cmap='viridis')
plt.matshow(x[0,:,:,100],cmap='viridis')
plt.matshow(x[0,:,:,110],cmap='viridis')
plt.matshow(x[0,:,:,112],cmap='viridis')
plt.matshow(x[0,:,:,113],cmap='viridis')
plt.matshow(x[0,:,:,114],cmap='viridis')
plt.matshow(x[0,:,:,115],cmap='viridis')
plt.matshow(x[0,:,:,116],cmap='viridis')
plt.matshow(x[0,:,:,117],cmap='viridis')
plt.matshow(x[0,:,:,118],cmap='viridis')
plt.matshow(x[0,:,:,119],cmap='viridis')
plt.matshow(x[0,:,:,120],cmap='viridis')
plt.matshow(x[0,:,:,121],cmap='viridis')
plt.matshow(x[0,:,:,122],cmap='viridis')
plt.matshow(x[0,:,:,123],cmap='viridis')
plt.matshow(x[0,:,:,122],cmap='viridis')
plt.matshow(x[0,:,:,123],cmap='viridis')
plt.matshow(x[0,:,:,124],cmap='viridis')
plt.matshow(x[0,:,:,125],cmap='viridis')
plt.matshow(x[0,:,:,126],cmap='viridis')
plt.matshow(x[0,:,:,127],cmap='viridis')
plt.matshow(x[0,:,:,128],cmap='viridis')
plt.matshow(x[0,:,:,129],cmap='viridis')
plt.show()
# for c in x[-1]:
#     plt.matshow(c,cmap='viridis')
#     plt.show()