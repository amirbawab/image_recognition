import keras
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam as Adam
from keras.layers.advanced_activations import LeakyReLU
import pandas as pd
import matplotlib.pyplot as plt


from keras import backend as K


EPOCH = 1
BATCH_SIZE = 128
alpha = 0.001 #learning rate
CLASS = 40
WIDTH = 64
HEIGHT = 64

def deepCNN():
    #Define the model:
    model = Sequential()
    #First layer
    model.add(Conv2D(32, (3, 3), input_shape=[WIDTH,HEIGHT,1], padding='same', activation='relu', name='1') )
    #model.add(BatchNormalization(axis=1))
    #model.add(Conv2D(32, (3, 3), input_shape=[WIDTH,HEIGHT,1], padding='same'))
    #model.add(LeakyReLU(alpha=0.3))
    #model.add(BatchNormalization(axis=1))
    #model.add(Conv2D(32, (3, 3), padding='same'))
    #model.add(LeakyReLU(alpha=0.3) )
    #model.add(BatchNormalization(axis=1))
    #Pool
    model.add(MaxPooling2D(pool_size=(2, 2) ))
    model.add(Dropout(0.5))
    #3rd
    model.add(Conv2D(32, (3, 3), padding='same', name='2'))
    model.add(LeakyReLU(alpha=0.3) )
    model.add(BatchNormalization(axis=1))
    #model.add(Conv2D(32, (3, 3), padding='same'))
    #model.add(LeakyReLU(alpha=0.3) )
    #model.add(BatchNormalization(axis=1))
    #model.add(Conv2D(32, (3, 3), padding='same'))
    #model.add(LeakyReLU(alpha=0.3) )
    #model.add(BatchNormalization(axis=1))
    #model.add(Conv2D(32, (3, 3), padding='same'))
    #model.add(LeakyReLU(alpha=0.3) )
    #model.add(BatchNormalization(axis=1))
    #pool
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(64, (3, 3), padding= 'same', name='3'))
    model.add(LeakyReLU(alpha=0.3)) 
    model.add(BatchNormalization(axis=1))
    #model.add(Conv2D(64, (3, 3), padding= 'same'))
    #model.add(LeakyReLU(alpha=0.3) )
    #model.add(BatchNormalization(axis=1))
    #model.add(Conv2D(64, (3, 3), padding= 'same'))
    #model.add(LeakyReLU(alpha=0.3)) 
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Conv2D(64, (3, 3), padding= 'same'))
    #model.add(LeakyReLU(alpha=0.3)) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    #4th layer
    model.add(Conv2D(64, (3, 3), padding= 'same', name='4'))
    model.add(LeakyReLU(alpha=0.3)) 
    model.add(BatchNormalization(axis=1))
    #model.add(Conv2D(64, (3, 3), padding= 'same'))
    #model.add(LeakyReLU(alpha=0.3) )
    #model.add(BatchNormalization(axis=1))
    #model.add(Conv2D(64, (3, 3), padding= 'same'))
    #model.add(LeakyReLU(alpha=0.3)) 
    #model.add(Conv2D(64, (3, 3), padding= 'same'))
    #model.add(LeakyReLU(alpha=0.3)) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    #Flatten
    model.add(Flatten())
    #Fully connected
    model.add(Dense(512, activation='tanh', name='5'))
    #Dropout
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='tanh', name='6'))
    #Final output layer
    model.add(Dense(CLASS, activation ='softmax', use_bias = True, name='7'))
    model.summary()
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'] )

    return model



def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations

def display_activations(activation_maps):
    """
    (1, 26, 26, 32)
    (1, 24, 24, 64)
    (1, 12, 12, 64)
    (1, 12, 12, 64)
    (1, 9216)
    (1, 128)
    (1, 128)
    (1, 10)
    """
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        plt.imshow(activations, interpolation='None', cmap='jet')
        plt.axis('off')
        plt.savefig('7.pdf', bbox_inches='tight')
        plt.show()

def main(x, y):
    #Reshape x:
    model = deepCNN()
    model.fit(x, y, 
            validation_data = (tx, ty),
            shuffle = True, epochs = EPOCH, 
            batch_size = BATCH_SIZE, verbose = 1)
    test=x[0]
    input_shape=[WIDTH,HEIGHT,1]
    test = np.random.random(input_shape)[np.newaxis,:]
    a = get_activations(model, test, False, layer_name='7')
    print("Activation")
    print(a)
    display_activations(a)

if __name__ == '__main__':
    #load data
    file_x = "hello2"

    tx = np.genfromtxt(file_x, delimiter = ' ', skip_header = 0)
    
    ty = tx[:, 0]
    tx = tx[:, 2:]

    ty = ty.reshape(-1, 1)

    #one hot encode ty, test_y
    enc = OneHotEncoder()

    ty = enc.fit_transform(ty)

    ty = ty.toarray()
    
    tx = np.reshape(tx, (-1, WIDTH, HEIGHT, 1))

    print(tx.shape)
    print(ty.shape)

    print("___________________________Finished Loading data___________________________")
    main(tx, ty)
