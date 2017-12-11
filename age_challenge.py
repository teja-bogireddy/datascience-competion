import pylab
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils, generic_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from scipy.misc import imread
from scipy import ndimage, misc
from keras import backend as K

K.set_image_dim_ordering('th')
img_rows, img_cols = 32, 32

root_dir = os.path.abspath('.')
data_dir = os.path.join(root_dir, 'Data')

train = pd.read_csv(os.path.join(data_dir, 'Train', 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'Test', 'test.csv'))

store = []
for img_id in train.ID:
	image_path = os.path.join(data_dir, 'Train', 'Train', img_id)
	img = imread(image_path, flatten = True)
	img_resize = misc.imresize(img, (32, 32))
	img_resize = img_resize.astype('float32')
	store.append(img_resize)

test_store = []
for img_id in test.ID:
	image_path = os.path.join(data_dir, 'Test', 'Test', img_id)
	img = imread(image_path, flatten = True)
	img_resize = misc.imresize(img, (32, 32))
        img_resize = img_resize.astype('float32')
	test_store.append(img_resize)

train_set = np.stack(store)
train_set = train_set / 255.
train_set = np.array(train_set)
num_samples = len(train_set)

test_set = np.stack(test_store)
test_set = test_set / 255.
test_set = np.array(test_set)
test_samples = len(test_set)

print "Load Training Data"

label = np.ones((num_samples, ), dtype = int)
index = 0
for img_class in train.Class:
	if img_class == "YOUNG":
		label[index] = 0
	elif img_class == "MIDDLE":
		label[index] = 1
	elif img_class == "OLD":
		label[index] = 2
	index = index + 1

print "Label Training Data"

train_data = [train_set, label]
test_data = [test_set]

(train_img, train_lab) = (train_data[0], train_data[1])
(test_img) = (test_data[0])

train_set = np.zeros((num_samples, 1, img_rows, img_cols))
test_set = np.zeros((test_samples, 1, img_rows, img_cols))

for idx in xrange(num_samples):
	train_set[idx][0][:][:] = train_img[idx, :, :]

for idx in xrange(test_samples):
        test_set[idx][0][:][:] = test_img[idx, :, :]

train_lab = np_utils.to_categorical(train_lab, 3)

print "Build Model"

model = Sequential()
model.add(Conv2D(32, (3, 3),input_shape = (1, 32, 32)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(3))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(train_set, train_lab, validation_data=(train_set, train_lab), batch_size=20, epochs = 10, verbose=1)

out2 = model.predict(test_set)
classes = np.argmax(out2, axis=1)
print classes

df = pd.DataFrame(classes)
df.to_csv('example.csv')
