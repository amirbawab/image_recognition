import tensorflow as tf 
import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import gc

LAYER = 2    #Rightnow 2 layers
HEIGHT = 64
WIDTH = 64
FEATURE = HEIGHT*WIDTH 
POOL_H = HEIGHT/LAYER
POOL_W = WIDTH/LAYER
RGB = 1
CLASS = 40
EPOCH = 1
BATCH_SIZE = 128
LEARNING_RATE = 1e-4



def mini_batch(x, y, batchsize):
    data_size = x.shape[0]
    for i in range(0, data_size - batchsize+1, batchsize):
        ind=slice(i,i+batchsize)
    yield x[ind], y[ind]


def deepnn(x):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
        x: an input tensor with the dimensions (N_examples, 100*70)
    Returns:
        A tuple (y, keep_prob). y is a tensor of shape (N_examples, 40), with values
        equal to the logits of classifying the digit into one of 40 classes.
        keep_prob is a scalar placeholder for the probability of
        dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, HEIGHT, WIDTH, RGB])

    # First convolutional layer - maps one grayscale image to 128 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps :B/C we downpool 2 times with 2*2, so first convolution layer -> 14*14, then 7*7
    # -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([16 * 16 * 64, 1024])
        b_fc1 = bias_variable([1024])
    
        h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 4096 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, CLASS])
        b_fc2 = bias_variable([CLASS])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu = False)


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)




def main(tx, ty, test_x, test_y, file_t, mapping):
    # Create the model
    x = tf.placeholder(tf.float32, [None, FEATURE])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, CLASS])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
    
    with tf.name_scope('prediction'):
        prediction = tf.argmax(y_conv, 1)

    graph_location = "/tmp/cnngraph/"
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    train_accuracy=[]
    test_accuracy=[]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(EPOCH):
            for batch in mini_batch(tx, ty, BATCH_SIZE):
                xs,ys=batch
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={ x: xs, y_: ys, keep_prob: 1.0})
                    #print('step %d, training accuracy %g' % (i, train_accuracy))
                    test_accuracy = accuracy.eval(feed_dict={ x: test_x, y_: test_y, keep_prob: 1.0})
                    train_accuray.append(i, train_accuracy)
                    test_accuray.append(i, test_accuracy)
                    print("Epoch = " + str(i) + " train accuracy: " + str(train_accuracy) + ", test accuracy: " + str(test_accuracy) )

                train_step.run(feed_dict={x: xs, y_: ys, keep_prob: 0.5})
                                
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: test_x, y_: test_y, keep_prob: 1.0}))
    
        #Free up rams
        test_x = None
        test_y = None
        tx = None
        ty = None
    
        gc.enable()    #Automatic garbage collection

        #Load test_x
        test= np.genfromtxt(file_t, delimiter=',')
                
        #Since our input format is different: a[0] is label, a[1] is pixel, a[2]: NAN data=a[3:]
        test = test[1:]
        test = test[:,2:]
        best = prediction.eval(feed_dict = { x: test, keep_prob: 1} )
        
        
        #Remap the indice of one hot encoded labels to its original label:
        remap = lambda x: mapping[x]
        best = best.tolist()        
        best = [remap(indice) for indice in best]

        #Write to prediction file
        pred = pd.DataFrame(data=best)
        pred.index+=1
        pred.to_csv("cnnPreProcessed_epoch1.csv", sep=',', header=['Label'], index=True, index_label='ID', encoding='utf-8')
        
        #Write train and test accuracy to a file:
        with open('train_accuracy.txt', 'w') as f:
            for train in train_accuracy:
                f.write(str(train[0]) + ', ' + str(train[1]) +'\n')
        
        with open('test_accuracy.txt', 'w') as f:
            for te in test_accuracy:
                f.write(str(te[0]) + ', ' + str(te[1]) +'\n')



if __name__ == '__main__':
    #Loading data
    file_x = "../data/trainx.csv"
    file_t = "../data/test.csv"

    tx= np.genfromtxt(file_x, delimiter=' ', skip_header=1)

    #test= np.genfromtxt(file_t, delimiter=',')

    ty = tx[:, 0]

    tx = tx[:, 2:]

    #Split train and test
    ind = int( tx.shape[0]/5*4 )
    test_x = tx[ind:]
    test_y = ty[ind:]
    test_y = test_y.reshape(-1, 1)

    tx = tx[:ind]
    ty = ty[:ind]
    ty = ty.reshape(-1, 1)

    #One hot encode y, ty
    enc = OneHotEncoder()

    ty = enc.fit_transform(ty)
    test_y = enc.transform(test_y)

    ty = ty.toarray()
    test_y = test_y.toarray()
    
    print(tx.shape)
    print(test_x.shape)
    print(ty.shape)
    print(test_y.shape)

    #Create a mapping between indice in one hot encoded labels to actual label
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]

    ind = [i  for i in range(40)]

    mapping = dict()
    for i, l in zip(ind, labels):
        mapping[i] = l
 

    print("__________________________________Finished Loading data__________________________________")
    #tf.app.run(main=main)
    main(tx, ty, test_x, test_y, file_t, mapping)
