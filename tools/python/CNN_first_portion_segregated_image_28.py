'''
First portion of Network

Input: takes 28X28 images from MNIST and EMNIST data
Output: 12 size vetctor predicting the probability of image belonging to 12 classes <0,1...9, A,M>
'''

from __future__ import print_function
import keras
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import ipdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 12
epochs = 10

# input image dimensions
img_rows, img_cols = 28,28

def load_data(path, val_perc = 0.1):
  data_txt = np.genfromtxt(os.path.join(path, 'all_ocv.ocv'), dtype=np.int32, delimiter=" ", skip_header=1)
  nSamples = data_txt.shape[0]
  np.random.shuffle(data_txt)
  x = data_txt[:,2:].reshape(-1,img_rows,img_cols)
  y = data_txt[:,0] 
  valLim = int(val_perc*nSamples)
  x_train = x[valLim:]
  y_train = y[valLim:]
  x_test = x[:valLim]
  y_test = y[:valLim]
  return (x_train,y_train), (x_test,y_test)

def load_test_prediction():
  data_txt = np.genfromtxt('test.ocv', dtype=np.int32, delimiter=" ", skip_header=1)
  nSamples = data_txt.shape[0]
  x = data_txt[:,2:].reshape(-1,img_rows,img_cols)
  return x

def WriteTestLabels(predicted_y, mapping_81, file_name):
  total_size = predicted_y.size
  print("Total images test data: ", str(total_size))
  data_labels = []
  for i in range(total_size):
    print(predicted_y[i])
    print(mapping_81[int(predicted_y[i])])
    data_labels.append(mapping_81[int(predicted_y[i])])

  with open(file_name, "w") as f:
    f.write("Id,Label")
    for i in range(10000):
      f.write("\n")
      f.write("{0},{1}".format(str(i+1), str(int(data_labels[i]))))

  print("Done writing labels in Test File")

def AddRandomNoise(images):
  #adds noise ranging from 1 to 255
  #max noise added to 10% of image

  num_images = images.shape[0]
  size_im = images.shape[1]
  images = images.reshape(num_images, size_im*size_im)
  for im_index in range(num_images):
    im_size = images[im_index].size
    max_noise = int(im_size * 0.25)
    min_noise = int(im_size * 0.15)
    num_elements_noise_added = np.random.randint(min_noise, max_noise)
    indexed_noisy_image=np.random.choice(im_size, num_elements_noise_added, replace=False)

    for i in indexed_noisy_image:
      images[im_index,i] = np.random.randint(0,255)

  images = images.reshape(num_images, size_im, size_im)

  return images

def PlotHistory(history):
  #history of accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig( "./accuracy.png")
  plt.close()
  
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig( "./loss.png")
  plt.close()

path = '/home/ml/ajain25/Documents/Courses/AML/Project_3/Keras/MNSIT_Data'
curr_path ='/home/ml/ajain25/Documents/Courses/AML/Project_3/Keras/MNSIT_Data/MNIST_rotated' 

if os.path.isfile(os.path.join(curr_path,'x_train.npy')):

  x_train = np.load(os.path.join(curr_path, 'x_train.npy'))
  y_train = np.load(os.path.join(curr_path, 'y_train.npy'))
  x_test = np.load(os.path.join(curr_path, 'x_test.npy'))
  y_test = np.load(os.path.join(curr_path, 'y_test.npy'))
else:
  (x_train, y_train), (x_test, y_test) = load_data(path)
  np.save(os.path.join(curr_path,'x_train'),x_train)
  np.save(os.path.join(curr_path,'y_train'),y_train)
  np.save(os.path.join(curr_path,'x_test'),x_test)
  np.save(os.path.join(curr_path,'y_test'),y_test)



#Add noise to train data
num_images_to_add_noise = int(x_train.shape[0] * 0.15)
noisy_images = x_train
x_train = AddRandomNoise(noisy_images)

#Add noise to test data
num_images_to_add_noise = int(x_test.shape[0] * 0.20)
noisy_images = x_test[:num_images_to_add_noise]
x_test[:num_images_to_add_noise] = AddRandomNoise(noisy_images)


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Model for first portion of network
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('CNN_FirstPortion_segregated_image.h5')
PlotHistory(history)

x_predict = load_test_prediction().astype('float32')

if K.image_data_format() == 'channels_first':
    x_predict = x_predict.reshape(x_predict.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_predict = x_predict.reshape(x_predict.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_predict /= 255

labels = model.predict(x_predict, batch_size=128)
predicted_labels = np.argmax(labels, axis=1)
WriteTestLabels(predicted_labels, mapping_81, "./TestPredicted.csv")


