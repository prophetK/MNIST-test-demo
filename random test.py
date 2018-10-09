import tensorflow as tf
import input_data
import numpy as np
import matplotlib.pyplot as plt
import pylab
import random

mnist = input_data.read_data_sets('data/', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


for i in range(0, len(mnist.test.images)):
  a = random.randint(0,len(mnist.test.images))
  result = sess.run(correct_prediction, feed_dict={x: np.array([mnist.test.images[a]]), y_: np.array([mnist.test.labels[a]])})
  if not result:
    print(sess.run(y, feed_dict={x: np.array([mnist.test.images[a]]), y_: np.array([mnist.test.labels[a]])}))
    print(sess.run(y_,feed_dict={x: np.array([mnist.test.images[a]]), y_: np.array([mnist.test.labels[a]])}))
    one_pic_arr = np.reshape(mnist.test.images[a], (28, 28))
    pic_matrix = np.matrix(one_pic_arr, dtype="float")
    plt.imshow(pic_matrix)
    pylab.show()
    break
#print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})