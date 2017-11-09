import keras
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam as Adam
from keras.layers.advanced_activations import LeakyReLU
import pandas as pd


EPOCH = 1000
BATCH_SIZE = 512
alpha = 0.001 #learning rate
CLASS = 40
WIDTH = 64
HEIGHT = 64

def deepCNN():
    #Define the model:
    model = Sequential()
    #First layer
    model.add(Conv2D(16, (5, 5), input_shape=[WIDTH,HEIGHT,1] , strides = 1, padding='same'))
    model.add(LeakyReLU(alpha=0.3) )
    model.add(BatchNormalization(axis=-1))
    #2nd
    model.add(Conv2D(16, (5, 5), strides = 1, padding='same'))
    model.add(LeakyReLU(alpha=0.3) )
    #Pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(BatchNormalization(axis=-1))
    #3rd
    model.add(Conv2D(32, (5, 5), strides = 1, padding='same'))
    model.add(LeakyReLU(alpha=0.3) )
    #pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    #4th
    model.add(Conv2D(32, (5, 5), strides = 1, padding='same'))
    model.add(LeakyReLU(alpha=0.3) )
    #pool
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    #Flatten
    model.add(Flatten())
    model.add(BatchNormalization(axis=-1))
    #Fully connected
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.3) )
    model.add(BatchNormalization(axis=-1))
    #Dropout
    model.add(Dropout(0.4))
    #Final output layer
    model.add(Dense(CLASS, activation ='softmax'))

    model.summary()

    model.compile(Adam(), loss = 'categorical_crossentropy', metrics=['accuracy'] )

    return model


def main(x, y, test_x, test_y, test, mapping):
    #Reshape x:
    model = deepCNN()
    model.fit(x, y, 
            validation_data = (tx, ty),
            shuffle = True, epochs = EPOCH, 
            batch_size = BATCH_SIZE, verbose = 2)
    
    model.save_weights('my_model_weights.h5')

    #Model prediction on testing data
    best = model.predict(test, batch_size = BATCH_SIZE)
    
    best = np.argmax(best, axis = 1) 
    
    #Remap the indice of one hot encoded labels to its original label:
    remap = lambda x: mapping[x]
    best = best.tolist()        
    best = [remap(indice) for indice in best]

    #Write to prediction file
    pred = pd.DataFrame(data=best)
    pred.index+=1
    pred.to_csv("cnn_KERAS_1000.csv", sep=',', header=['Label'], index=True, index_label='ID', encoding='utf-8')


if __name__ == '__main__':
    #load data
    file_x = "../data/newClean/train.csv"
    file_t = "../data/newClean/test.csv"

    tx = np.genfromtxt(file_x, delimiter = ' ', skip_header = 1)
    test = np.genfromtxt(file_t, delimiter = ' ', skip_header = 1)

    ty = tx[:, 0]
    tx = tx[:, 2:]

    test = test[:, 2:]
    
    #randomly shuffle tx:
    np.random.shuffle(tx)
    
    #Split train and test
    ind = int(tx.shape[0]/10*9.5)
    test_x = tx[ind:]
    test_y = ty[ind:]

    test_y = test_y.reshape(-1, 1)
    ty = ty.reshape(-1, 1)

    #one hot encode ty, test_y
    enc = OneHotEncoder()

    ty = enc.fit_transform(ty)
    test_y = enc.transform(test_y)

    ty = ty.toarray()
    test_y = test_y.toarray()
    
    tx = np.reshape(tx, (-1, WIDTH, HEIGHT, 1))
    test_x = np.reshape(test_x, (-1, WIDTH, HEIGHT, 1))
    test = np.reshape(test, (-1, WIDTH, HEIGHT, 1))

    #Create a mapping between indice in one hot encoded labels to actual label
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]

    ind = [i  for i in range(40)]

    mapping = dict()
    for i, l in zip(ind, labels):
        mapping[i] = l
    
    print(tx.shape)
    print(test_x.shape)
    print(test_y.shape)
    print(test.shape)

    print("___________________________Finished Loading data___________________________")
    main(tx, ty, test_x, test_y, test, mapping)
