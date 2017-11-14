'''
Using 12X3 vectors from forst portion of CNN architecture
gets output 40 class classification

Input: 3 images with same labels (segregated image with 2 digits and 1 alphabet)
Output: 40 class classification problem
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

num_classes = 40
batch_size = 128
epochs = 150

# input image dimensions
img_rows, img_cols = 28, 28

def LoadTrainData():
  path = '/home/ml/ajain25/Documents/Courses/AML/Project_3/NewDataMnsit'
  data_txt = np.genfromtxt(os.path.join(path,'train.ocv'), dtype=np.int32, delimiter=" ", skip_header=1)
  
  img = data_txt[:,2:]
  y = data_txt[:,0]
  return img, y


def LoadTestData():
  path = '/home/ml/ajain25/Documents/Courses/AML/Project_3/NewDataMnsit'
  data_txt = np.genfromtxt(os.path.join(path,'test.ocv'), dtype=np.int32, delimiter=" ", skip_header=1)
  x = data_txt[:,2:]

  return x
  

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
  print("Done writing labels in Test File")


def PlotHistory(history):
  #history of accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig( "./accuracy_mnist_extended_nn_5_mnist_noiseAdded_v1.png")
  plt.close()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig( "./loss_mnist_extended_nn_5_mnist_noiseAdded_v1.png")
  plt.close()

def SaveCombinedFeatureData(x_train, y_train, x_val, y_val, x_test):
  np.save('x_train',x_train)
  np.save('y_train',y_train)
  np.save('x_val',x_val)
  np.save('y_val',y_val)
  np.save('x_test',x_test)

def GetMappingTo40(mapping, labels):
	y=[]
	for i in labels:
	  y.append(mapping[i])
	y_mappedto_40 = np.array(y).astype('int32')

	return y_mappedto_40

def GetPredictedFeaturesFromMNIST(data, model):
	img1 = data[0::3, :]
	img2 = data[1::3, :]
	img3 = data[2::3, :]

	if K.image_data_format() == 'channels_first':
		img1 = img1.reshape(img1.shape[0], 1, img_rows, img_cols)
		img2 = img2.reshape(img2.shape[0], 1, img_rows, img_cols)
		img3 = img3.reshape(img3.shape[0], 1, img_rows, img_cols)
	else:
		img1 = img1.reshape(img1.shape[0], img_rows, img_cols, 1)
		img2 = img2.reshape(img2.shape[0], img_rows, img_cols, 1)
		img3 = img3.reshape(img3.shape[0], img_rows, img_cols, 1)

	p_label1 = model.predict(img1, batch_size=batch_size)
	p_label2 = model.predict(img2, batch_size=batch_size)
	p_label3 = model.predict(img3, batch_size=batch_size)

	features = np.hstack((p_label1, p_label2, p_label3))

	return features


#Loading the segregated inages data from train and test
x_train, y_train = LoadTrainData()
x_test = LoadTestData()

#learning mapping from 81 classes to 40 labels
labels_global = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
mapping_40 = {}
mapping_81 = {}
for i,l in enumerate(labels_global):
  mapping_40[l] =i
  mapping_81[i] = l

#Loaded already trained EMNIST model
model = load_model('/home/ml/ajain25/Documents/Courses/AML/Project_3/Keras/MNSIT_Data/MNIST_rotated/my_model_EMNIST_Rotated_9_v4_noise_added_all_data.h5')
y_train = GetMappingTo40(mapping_40, y_train)

#normalizing the images
x_train = x_train/ 255.0
data_features = GetPredictedFeaturesFromMNIST(x_train, model)

#because label is same for all 3 images
y = y_train[0::3]
indices = np.random.permutation(y.size)
data_features = data_features[indices]
y = y[indices]

#divide into validation data
val_prec = 0.2 
val_limits = int(val_prec * y.size)

x_val = data_features[:val_limits, :]
x_train = data_features[val_limits: , :]

y_val = y[:val_limits]
y_train = y[val_limits:]

# #test data
x_test = x_test/ 255.0
x_test = GetPredictedFeaturesFromMNIST(x_test, model)

SaveCombinedFeatureData(x_train, y_train, x_val, y_val, x_test)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
input_shape = x_train.shape[1]

#Adding more layer to predict 12*3 classes to 40 classes (final labels)
model = Sequential()
model.add(Dense(512, input_dim=36, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, input_dim=36, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adamax(decay= 1e-4),
              metrics=['accuracy'])
history_nn = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val)) 
model.save("CNN_second_portion.h5")
PlotHistory(history_nn)


y_predicted_test = model.predict(x_test)
predicted_labels = np.argmax(y_predicted_test, axis=1)

print("size of labels: ", predicted_labels.shape)
WriteTestLabels(predicted_labels, mapping_81, "./TestPredicted.csv")