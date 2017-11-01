import tensorflow as tf
import numpy as np

file_x = "x.csv"
file_y = "y.csv"
train_x= np.genfromtxt(file_x, delimiter=',')
train_y= np.genfromtxt(file_y, delimiter=',')

tx = tf.convert_to_tensor(train_x, dtype = tf.float32)
ty = tf.convert_to_tensor(train_y, dtype = tf.float32)

print("Finished loading data")

#sess = tf.Session()

x = tf.placeholder(tf.float32, [None, 4096])     #here none means it can hold any rows. 

W = tf.Variable(tf.zeros([4096, 40]))
b = tf.Variable(tf.zeros([40]))

#evidence is z = xW+b
y = tf.matmul(x, W) + b

#Training
#y_ to hold prediction
y_ = tf.placeholder(tf.float32, [None, 40])

#loss function (Cross-entropy)
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#cross_entropy2 = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(s), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

print("Finished defining variables and functions")

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

print("Trying batch")
#Prepare batches
BATCH_SIZE = 128
capacity = 10*BATCH_SIZE

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

[batch_xs, batch_ys] = tf.train.batch([tx, ty], batch_size = BATCH_SIZE, capacity = capacity, enqueue_many=True)
print("Trying to run batch")
sess.run([batch_xs, batch_ys])
print(batch_ys)

'''
for _ in range(2):
    [batch_xs, batch_ys] = tf.train.batch([tx, ty], batch_size = BATCH_SIZE, capacity = capacity, enqueue_many=True)
    sess.run([batch_xs, batch_ys])
    print(batch_ys)
    coord.join(threads)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


#Calculate correct predictions
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
'''
