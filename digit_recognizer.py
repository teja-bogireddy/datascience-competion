import pandas as pd
import numpy as np
import keras
from keras.utils import np_utils, generic_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from scipy.misc import imread
from scipy import ndimage, misc
from keras import backend as K

K.set_image_dim_ordering('th')
img_rows, img_cols = 28, 28

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
train_lab = (train_data.ix[:,0].values).astype('int32')
train_pix = (train_data.ix[:,1:].values).astype('float32')
test_pix = (test_data.ix[:,:].values).astype('float32')

train_pix = train_pix.reshape(train_pix.shape[0], 28, 28)
test_pix = test_pix.reshape(test_pix.shape[0], 28, 28)

train_samples = train_pix.shape[0]
test_samples = test_pix.shape[0]

train_lab = to_categorical(train_lab)

train_set = np.zeros((train_samples, 1, img_rows, img_cols))
test_set = np.zeros((test_samples, 1, img_rows, img_cols))

train_pix = train_pix/255.
test_pix = test_pix/255.

for idx in xrange(train_samples):
	train_set[idx][0][:][:] = train_pix[idx, :, :]

for idx in xrange(test_samples):
	test_set[idx][0][:][:] = test_pix[idx, :, :]

model = Sequential()
model.add(Conv2D(32, (3, 3),input_shape = (1, 28, 28)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(train_set, train_lab, validation_data=(train_set, train_lab), batch_size=20, epochs = 5, verbose=1)

out2 = model.predict(test_set)
classes = np.argmax(out2, axis = 1)

print classes

df = pd.DataFrame(classes)
df.to_csv('prediction.csv')
