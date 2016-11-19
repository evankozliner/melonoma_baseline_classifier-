'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import numpy as np
import csv
from logistic_regression_classifier import plot_confusion_matrix

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 5
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes
#record = tf.Variable(tf.float32, [mnist.train.num_examples, 1,1])

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax


# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c,prediction = sess.run([optimizer, cost,pred], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            #print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
            print("Epoch " + str(epoch) + " cost: " + str(avg_cost))

    print("Optimization Finished!")

    # Test model

    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    #prediction = tf.argmax(pred,1)
    #print(prediction)
    #true_value = tf.argmax(y,1)
    #correct_prediction = tf.equal(prediction, true_value)
    # Calculate accuracy
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print("Accuracy:" + str(accuracy.eval({x: mnist.test.images, y: mnist.test.labels})))
    output_file = open("out.csv", 'a')
    writer = csv.writer(output_file)
    for i in range(mnist.test.num_examples):
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c,prediction = sess.run([optimizer, cost,pred], feed_dict={x: batch_xs,
                                                      y: batch_ys})
        
        for i in range(prediction.shape[0]):
            writer.writerow([prediction[i].tolist().index(np.max(prediction[i])), 
                batch_ys[i].tolist().index(1.0)])

        #print (tf.equal(tf.argmax(pred[0],1), tf.argmax(batch_ys,1)))
        # Compute average loss
        avg_cost += c / total_batch
    output_file.close()
