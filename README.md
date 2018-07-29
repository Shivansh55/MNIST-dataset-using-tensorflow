# MNIST-dataset-using-tensorflow
MNIST is a simple computer vision dataset. It consists of images of handwritten digits
I have implemented it using tensorflow
The MNIST data is split into three parts: 55,000 data points of training data (mnist.train), 10,000 points of test data (mnist.test), and 5,000 points of validation data (mnist.validation).
MNIST data point has two parts: an image of a handwritten digit and a corresponding label. We'll call the images "x" and the labels "y". Both the training set and test set contain images and their corresponding labels; for example the training images are mnist.train.images and the training labels are mnist.train.labels.
We can flatten this array into a vector of 28x28 = 784 numbers. It doesn't matter how we flatten the array, as long as we're consistent between images. From this perspective, the MNIST images are just a bunch of points in a 784-dimensional vector space
I have used one-hot encoding vectors since the result will be between 0-9(digits).Therefore one hot vector will be zero in most dimensions and 1 in one dimension.An example would [0,0,0,0,0,0,0,1,0,0]
Softmax regression is best suited for MNIST data set as the probabalities that it belongs to different classes add up to 1.If the probability that it belongs to a particuylar class is high then the corresponding weight is exponentially increased and vice versa for lower probability.Bias is added because there are some independent parameters that control the result.
Loss is calculated using cross entropy which shows how much our model is deviated from predicting the correct outputs
Gradient descent is used for updating the weights using GradientDescentOptimizer
The accuracy comes to be around 92%
In the next project I will work upon MNIST using deep CNN to get better results.



