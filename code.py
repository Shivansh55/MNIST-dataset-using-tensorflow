#MNIST is a computer vision dataset consisting of images of handwritten digits

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#array is flattened into a vector of 784=28*28 
#mnist.train.images is a tensor (an n-dimensional array) with a shape of [55000, 784]

#entry in the tensor is a pixel intensity between 0 and 1
#which is due to normalization(all are divided by 255)

import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#W has a shape of [784, 10] because we want to multiply the 784-dimensional image vectors
#to produce 10-dimensional vectors of evidence for the difference classes. 
#b has a shape of [10] so we can add it to the output.

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#tf.reduce_sum adds the elements in the second dimension of y
# due to the reduction_indices=[1] parameter

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)#learning rate of 0.5
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
