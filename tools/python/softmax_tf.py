import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

#Loading data
file_x = "x.csv"
file_y = "y.csv"
file_t = "test.csv"

tx= np.genfromtxt(file_x, delimiter=',')
ty= np.genfromtxt(file_y, delimiter=',')
test= np.genfromtxt(file_t, delimiter=',')

#Split train and test
ind = int( tx.shape[0]/5*4 )
test_x = tx[ind:]
test_y = ty[ind:]
test_y = test_y.reshape(test_y.shape[0], 1)

tx = tx[:ind]
ty = ty[:ind]
ty = ty.reshape(ty.shape[0], 1)

print("Finished loading data")

#Define params:
nb_classes = 40
learn_rate = 0.001
BATCH_SIZE = 256
epoch = 30000
log_file = "/tmp/mini_Softmax1.log"
data_size = ty.shape[0]


#One hot encode y, ty
enc = OneHotEncoder()

ty = enc.fit_transform(ty)
test_y = enc.transform(test_y)

ty = ty.toarray()
test_y = test_y.toarray()


#Define tensorflow variables
x = tf.placeholder(tf.float32, [None, 4096])     #here none means it can hold any rows. 

W = tf.Variable(tf.zeros([4096, 40]))
b = tf.Variable(tf.zeros([40]))

#evidence is z = xW+b
y = tf.matmul(x, W) + b

#Training
#y_ to hold prediction
y_ = tf.placeholder(tf.float32, [None, 40])

#tensorflow summary to collect stats
W_hist = tf.summary.histogram("weights", W)
b_hist = tf.summary.histogram("biases", b)
y_hist = tf.summary.histogram("y", y)


#loss function (Cross-entropy)
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#cross_entropy2 = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(s), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)



sess = tf.Session()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_file, sess.graph_def)


init = tf.global_variables_initializer()

sess.run(init)

#define batch

def mini_batch(x, y, batchsize):
    for i in range(0, data_size - batchsize+1, batchsize):
	    ind=slice(i,i+batchsize)
	    yield x[ind], y[ind]


for i in range(epoch):
    for batch in mini_batch(tx, ty, BATCH_SIZE):
        xs,ys=batch
        #xs =  tf.convert_to_tensor(xs, dtype = tf.float32)
        #ys =  tf.convert_to_tensor(ys, dtype = tf.float32)
  
        if i%10 == 0:
            all_feed = { x: tx, y_: ty }
            result = sess.run(merged, feed_dict = all_feed)
            writer.add_summary(result, i)
        else:
            feed = { x:xs, y_:ys}
            sess.run(train_step, feed_dict = feed)


#Calculate correct predictions
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
feed = { x : test_x , y_: test_y  }
print(sess.run(accuracy, feed_dict=feed))


prediction = tf.argmax(y,1)
best = sess.run(prediction, feed_dict = { x: test })


#Write to prediction file
pred = pd.DataFrame(data=best)
pred.to_csv("Softmax_pred.csv", sep=',', index=False, encoding='utf-8')

