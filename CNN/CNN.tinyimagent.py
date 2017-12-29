"""
Authored by: Hager Radi
16 Nov. 2016
A basic implementation of a CNN model in Python, trained on TinyImageNet
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os
import sys
import re
import numpy as np
import Image
from os import listdir
from os.path import isfile, join
import matplotlib.image as img
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2
#mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)


train_dir = '/home/hager/Documents/tiny-imagenet/train'
validation_dir = '/home/hager/Documents/tiny-imagenet/validation'

n_classes = 100
batch_size = 128

x = tf.placeholder('float', [12288, None])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,3,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([16*16*64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 64, 64, 3])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 16*16*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    images, labels = read_training_set()
    prediction = convolutional_neural_network(x)
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    cost = tf.reduce_mean( tf.nn.softmax(prediction) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    num_examples = 50000
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(int(num_examples)):
                epoch_x, epoch_y = images[i] , labels[i]
                #print type(epoch_x)
                epoch_x = epoch_x.reshape(12288,1)
                print np.shape(epoch_x)
                print np.shape(epoch_y)
                i, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        x_val , y_val = GetValset
        print('Accuracy:',accuracy.eval({x:x_val, y:y_val}))

def read_training_set():
        files = os.listdir(train_dir)
        class_paths = []
        data_labels = []
        data_images = np.zeros(shape=(50000,64,64,3))
        for i in range(len(files)):
            class_path = train_dir + '/' + files[i] + '/images/'
            images = os.listdir(class_path)
            for im in range(len(images)):
                img = cv2.imread(class_path+images[im])
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_mat = np.array(image)
                data_images[i*im] = np.array(image_mat)
                data_labels.append(np.array(i))

        data_images = data_images.reshape((50000, 3*64*64))
        print np.shape(data_labels)
        print np.shape(data_images)

        return data_images,data_labels
def read_validation_set():
    files = os.listdir(validation_dir)
    class_paths = []
    data_labels = []
    data_images = np.zeros(shape=(50000,64,64,3))
    for i in range(len(files)):
        class_path = validation_dir + '/' + files[i] + '/images/'
        images = os.listdir(class_path)
        for im in range(len(images)):
            img = cv2.imread(class_path+images[im])
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_mat = np.array(image)
            data_images[i*im] = np.array(image_mat)
            data_labels.append(np.array(i))

    data_images = data_images.reshape((50000, 3*64*64))
    print np.shape(data_labels)
    print np.shape(data_images)

    return data_images,data_labels


def main():
    train_neural_network(x)

if __name__=='__main__':
    main()
