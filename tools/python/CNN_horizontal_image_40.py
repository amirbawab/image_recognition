'''
CNN architecture that takes 40X40 image as input and generates 40
class classification labels as output.

Input: Space separated file with labels and 40X40 size vector for image
Output: Generates a CSV for test labels provided to us
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
from keras import metrics
import ipdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

batch_size = 80
num_classes = 40
epochs = 10

# input image dimensions
img_rows, img_cols = 40, 40

 
def LoadTrainData(shuffle = True, val_perc = 0.2):
  path = '/home/ml/ajain25/Documents/Courses/AML/Project_3/CNN_horizontal_image/40_size'
  data_txt = np.genfromtxt(os.path.join(path,'train.ocv'), dtype=np.int32, delimiter=" ", skip_header=1)
  if shuffle == True:
    np.random.shuffle(data_txt)

  nSamples = data_txt.shape[0]
  x = data_txt[:,2:].reshape(-1,img_rows,img_cols)
  y = data_txt[:,0] 
  valLim = int(val_perc*nSamples)
  x_train = x[valLim:]
  y_train = y[valLim:]
  x_test = x[:valLim]
  y_test = y[:valLim]
  
  return (x_train,y_train), (x_test,y_test)


def LoadTestData():
  path = '/home/ml/ajain25/Documents/Courses/AML/Project_3/CNN_horizontal_image/40_size'
  data_txt = np.genfromtxt(os.path.join(path,'test.ocv'), dtype=np.int32, delimiter=" ", skip_header=1)
  nSamples = data_txt.shape[0]
  x = data_txt[:,2:].reshape(-1,img_rows,img_cols)

  return x
  
# Writes the testing data into csv format <kaggle csv format>
def WriteTestLabels(predicted_y, mapping_81, file_name):
  total_size = predicted_y.size
  print("Total images test data: ", str(total_size))
  data_labels = []
  for i in range(total_size):
    data_labels.append(mapping_81[int(predicted_y[i])])

  with open(file_name, "w") as f:
    f.write("Id,Label")
    for i in range(10000):
      f.write("\n")
      f.write("{0},{1}".format(str(i+1), str(int(data_labels[i]))))

def PlotHistory(history):

  #accuracy graph
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


#if numpy exist then load it else create from csv file
if os.path.isfile('x_train.npy'):
  x_train = np.load('x_train.npy')
  y_train = np.load('y_train.npy')
  x_val = np.load('x_test.npy')
  y_val = np.load('y_test.npy')

else:
  (x_train, y_train), (x_val, y_val) = LoadTrainData()
  np.save('x_train',x_train)
  np.save('y_train',y_train)
  np.save('x_test',x_val)
  np.save('y_test',y_val)

#generates the test file
if os.path.isfile('testdata.npy'):
  x_test = np.load('testdata.npy')
else:
  x_test = LoadTestData()
  np.save('testdata', x_test)

#mapping of labels from <0.. 81> to 40 class
labels_global = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
mapping_40 = {}
mapping_81 = {}
for i,l in enumerate(labels_global):
  mapping_40[l] =i
  mapping_81[i] = l

#Mapping training data to 40 class 
y=[]
for i in y_train:
  y.append(mapping_40[i])
y_train = np.array(y).astype('int32')

#Mapping validation data to 40 class 
y=[]
for i in y_val:
  y.append(mapping_40[i])
y_val = np.array(y).astype('int32')


if K.image_data_format() == 'channels_first':
  x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
  x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
  input_shape = (1, img_rows, img_cols)
else:
  x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
  x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)

#image normalization
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_val /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'val samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)


# model = load_model('my_model_11_nov.h5')
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1e-5, decay=1e-6),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val))

score = model.evaluate(x_val, y_val, verbose=0)
print('val loss:', score[0])
print('val accuracy:', score[1])

model.save('CNN_model_Horizontal.h5')

PlotHistory(history)

#predicting test data 
x_predict = x_test.astype('float32')

if K.image_data_format() == 'channels_first':
    x_predict = x_predict.reshape(x_predict.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_predict = x_predict.reshape(x_predict.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_predict /= 255

labels = model.predict(x_predict, batch_size=128)
predicted_labels = np.argmax(labels, axis=1)

print("size of labels: ", labels.shape)
WriteTestLabels(predicted_labels, mapping_81, "./TestPredicted.csv")