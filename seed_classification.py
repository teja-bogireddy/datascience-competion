import os
import numpy as np
import pandas as pd
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

root_directory = os.path.abspath('.')
train_directory = os.path.join(root_directory, 'train')
blackgrass_directory = os.path.join(train_directory,'Black-grass')
blackgrasslis_directory = os.listdir(blackgrass_directory)
charlock_directory = os.path.join(train_directory,'Charlock')
charlocklis_directory = os.listdir(charlock_directory)
cleavers_directory = os.path.join(train_directory,'Cleavers')
cleaverslis_directory = os.listdir(cleavers_directory)
commonchickweed_directory = os.path.join(train_directory,'Common Chickweed')
commonchickweedlis_directory = os.listdir(commonchickweed_directory)
commonwheat_directory = os.path.join(train_directory,'Common wheat')
commonwheatlis_directory = os.listdir(commonwheat_directory)
fathen_directory = os.path.join(train_directory,'Fat Hen')
fathenlis_directory = os.listdir(fathen_directory)
loosesilkybent_directory = os.path.join(train_directory,'Loose Silky-bent')
loosesilkybentlis_directory = os.listdir(loosesilkybent_directory)
maize_directory = os.path.join(train_directory,'Maize')
maizelis_directory = os.listdir(maize_directory)
scentlessmayweed_directory = os.path.join(train_directory,'Scentless Mayweed')
scentlessmayweedlis_directory = os.listdir(scentlessmayweed_directory)
shepherdspurse_directory = os.path.join(train_directory,'Shepherds Purse')
shepherdspurselis_directory = os.listdir(shepherdspurse_directory)
smallfloweredcranesbill_directory = os.path.join(train_directory,'Small-flowered Cranesbill')
smallfloweredcranesbilllis_directory = os.listdir(smallfloweredcranesbill_directory)
sugarbeet_directory = os.path.join(train_directory,'Sugar beet')
sugarbeetlis_directory = os.listdir(sugarbeet_directory)

train = []
indx = 0
print indx
for image in blackgrasslis_directory:
	image_path = os.path.join(blackgrass_directory, image)
	image_pix = imread(image_path,flatten = True)
	image_pix = misc.imresize(image_pix, (32, 32))
	image_pix = image_pix.astype('float32')
	train.append(image_pix)
	indx = indx + 1
print indx
for image in charlocklis_directory:
        image_path = os.path.join(charlock_directory, image)
        image_pix = imread(image_path,flatten = True)
        image_pix = misc.imresize(image_pix, (32, 32))
        image_pix = image_pix.astype('float32')
        train.append(image_pix)
	indx = indx + 1
print indx

for image in cleaverslis_directory:
        image_path = os.path.join(cleavers_directory, image)
        image_pix = imread(image_path,flatten = True)
        image_pix = misc.imresize(image_pix, (32, 32))
        image_pix = image_pix.astype('float32')
        train.append(image_pix)
	indx = indx + 1
print indx

for image in commonchickweedlis_directory:
        image_path = os.path.join(commonchickweed_directory, image)
        image_pix = imread(image_path,flatten = True)
        image_pix = misc.imresize(image_pix, (32, 32))
        image_pix = image_pix.astype('float32')
        train.append(image_pix)
	indx = indx + 1
print indx

for image in commonwheatlis_directory:
        image_path = os.path.join(commonwheat_directory, image)
        image_pix = imread(image_path,flatten = True)
        image_pix = misc.imresize(image_pix, (32, 32))
        image_pix = image_pix.astype('float32')
        train.append(image_pix)
	indx = indx + 1
print indx

for image in fathenlis_directory:
        image_path = os.path.join(fathen_directory, image)
        image_pix = imread(image_path,flatten = True)
        image_pix = misc.imresize(image_pix, (32, 32))
        image_pix = image_pix.astype('float32')
        train.append(image_pix)
	indx = indx + 1
print indx

for image in loosesilkybentlis_directory:
        image_path = os.path.join(loosesilkybent_directory, image)
        image_pix = imread(image_path,flatten = True)
        image_pix = misc.imresize(image_pix, (32, 32))
        image_pix = image_pix.astype('float32')
        train.append(image_pix)
	indx = indx + 1
print indx

for image in maizelis_directory:
        image_path = os.path.join(maize_directory, image)
        image_pix = imread(image_path,flatten = True)
        image_pix = misc.imresize(image_pix, (32, 32))
        image_pix = image_pix.astype('float32')
        train.append(image_pix)
	indx = indx + 1
print indx

for image in scentlessmayweedlis_directory:
        image_path = os.path.join(scentlessmayweed_directory, image)
        image_pix = imread(image_path,flatten = True)
        image_pix = misc.imresize(image_pix, (32, 32))
        image_pix = image_pix.astype('float32')
        train.append(image_pix)
	indx = indx + 1
print indx

for image in shepherdspurselis_directory:
        image_path = os.path.join(shepherdspurse_directory, image)
        image_pix = imread(image_path,flatten = True)
        image_pix = misc.imresize(image_pix, (32, 32))
        image_pix = image_pix.astype('float32')
        train.append(image_pix)
	indx = indx + 1
print indx

for image in smallfloweredcranesbilllis_directory:
        image_path = os.path.join(smallfloweredcranesbill_directory, image)
        image_pix = imread(image_path,flatten = True)
        image_pix = misc.imresize(image_pix, (32, 32))
        image_pix = image_pix.astype('float32')
        train.append(image_pix)
	indx = indx + 1
print indx

for image in sugarbeetlis_directory:
        image_path = os.path.join(sugarbeet_directory, image)
        image_pix = imread(image_path,flatten = True)
        image_pix = misc.imresize(image_pix, (32, 32))
        image_pix = image_pix.astype('float32')
        train.append(image_pix)
	indx = indx + 1
print indx

test_directory = os.path.join(root_directory, 'test')
testlis_directory = os.listdir(test_directory)
print testlis_directory
test = []
for image in testlis_directory:
	image_path = os.path.join(test_directory, image)
        image_pix = imread(image_path,flatten = True)
        image_pix = misc.imresize(image_pix, (32, 32))
        image_pix = image_pix.astype('float32')
        test.append(image_pix)

train_set = np.stack(train)
train_set = train_set / 255.
train_set = np.array(train_set)
train_samples = len(train_set)

test_set = np.stack(test)
test_set = test_set / 255.
test_set = np.array(test_set)
test_samples = len(test_set)

label = np.ones((train_samples, ), dtype = int)
label[:262] = 0
label[263:652] = 1
label[653:939] = 2
label[940:1550] = 3
label[1551:1771] = 4
label[1772:2246] = 5
label[2247:2900] = 6
label[2901:3121] = 7
label[3122:3637] = 8
label[3638:3868] = 9
label[3869:4364] = 10
label[4365:4749] = 11

train_data = [train_set, label]
test_data = [test_set]

(train_img, train_lab) = (train_data[0], train_data[1])
(test_img) = (test_data[0])

train_set = np.zeros((train_samples, 1, img_rows, img_cols))
test_set = np.zeros((test_samples, 1, img_rows, img_cols))

for idx in xrange(train_samples):
	train_set[idx][0][:][:] = train_img[idx, :, :]

for idx in xrange(test_samples):
        test_set[idx][0][:][:] = test_img[idx, :, :]

train_lab = np_utils.to_categorical(train_lab, 12)

model = Sequential()
model.add(Conv2D(32, (3, 3),input_shape = (1, 32, 32)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(12))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(train_set, train_lab, validation_data=(train_set, train_lab), batch_size=20, epochs = 18, verbose=1)

out2 = model.predict(test_set)
classes = np.argmax(out2, axis=1)
print classes

df = pd.DataFrame(testlis_directory,classes)
df.to_csv('prediction.csv')

