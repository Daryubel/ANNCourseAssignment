import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

from PIL import Image

import tensorflow.keras as keras
from tensorflow.keras import layers


def create_data(path):
    pics = os.listdir(path)
    pics.sort(key=lambda x: int(x.split('.')[0]))
    data = []
    for item in pics:
        data.append(plt.imread(path+'\\'+item).tolist())
    return data


def create_set(hf, name, data):
    hf.create_dataset(name, data=data)


# raw figure resizing and import
for i in range(80):
    i = '%d' % (i+1)
    str0 = '.\\png\\' + i + '.png'
    Dim = Image.open(str0)
    # Dim.show()
    ZoomDim = Dim.resize((64, 64))
    ZoomDim = ZoomDim.convert('L')
    str1 = '.\\png_unity\\' + i + '.png'
    ZoomDim.save(str1)

# raw figure conversion to hdf5 files
f0 = h5py.File('.\\h5\\data_train.h5', 'w')
all_data = create_data('.\\png_unity')
create_set(f0, 'train_set_1', all_data)
f0.close()
# hdf5 files import as datasets
f1 = h5py.File('.\\h5\\data_train.h5', 'r')
train_data = f1['train_set_1'][:]
# labeling
train_label = np.zeros(80,)
for i in range(80):
    if 0 <= i < 10:
        train_label[i, ] = 0
    elif 10 <= i < 20:
        train_label[i, ] = 1
    elif 20 <= i < 30:
        train_label[i, ] = 2
    else:
        train_label[i, ] = 3
# perceptron
model = keras.Sequential()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))
# hyper parameters
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
# train
model.fit(train_data, train_label, epochs=200, batch_size=128)
model.evaluate(train_data, train_label)
# model save
model.save('.\\h5\\model_1.h5')

# predict with saved model
f2 = h5py.File('.\\h5\\data_predict.h5', 'w')
all_data = create_data('.\\predict')
create_set(f2, 'train_predict_1', all_data)
f2.close()
f3 = h5py.File('.\\h5\\data_predict.h5', 'r')
train_predict = f3['train_predict_1'][:]
predict_x = model.predict(train_predict)
classes_x = np.argmax(predict_x, axis=1)
for i in classes_x:
    if i == 0:
        print('丁')
    elif i == 1:
        print('小')
    elif i == 2:
        print('林')
    else:
        print('其他')
