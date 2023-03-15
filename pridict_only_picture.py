import numpy as np
import	cv2
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession




config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



model = load_model('D:/python/sheep_face/edge_face_alxe_weight/weights_195-0.1644-0.91842.h5', compile=None)
data = 'D:/python/sheep_face/lian_data/train1/sheep2/019 039_1.jpg'
data = image.load_img(data)




data_gray = data.convert('L')
x = image.img_to_array(data_gray)
x = cv2.resize(x, (224, 224))
x = x/255#要对图片进行预处理，其操作应与原来的训练集的操作一致


# 扩充图片的维度【数量，（图片大小），通道】
x = np.expand_dims(x, axis=0)
x = np.expand_dims(x, axis=3)



print(x.shape)
plt.imshow(data)


plt.show()
predict = model.predict(x)
predict = np.argmax(predict, axis=1)
print(predict)