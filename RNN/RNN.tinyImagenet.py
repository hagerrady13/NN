"""
Authored by: Hager Radi
16 Nov. 2016
A basic implementation of an RNN model in Python, trained on TinyImageNet
"""
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

import tensorflow as tf
import os
import sys
import re
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.image as img
from skimage.io import imread
import cv2

from datetime import datetime
import glob
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
import struct

FLAGS = tf.app.flags.FLAGS

x = tf.placeholder('float', [None,n_chunks,chunk_size])
y = tf.placeholder('float')

tf.app.flags.DEFINE_string('output_graph', '/home/hagerradi/rnn_output_graph.pb',"")
train_dir = '/home/hagerradi/tiny-imagenet/train'
validation_dir = '/home/hagerradi/tiny-imagenet/validation'

hm_epochs = 1
n_classes = 100
batch_size = 500   # 1000
chunk_size = 1536
n_chunks = 8
rnn_size = 512
number_of_layers = 3

def read_training_set():
        files = os.listdir(train_dir)
        class_paths = []
        data_labels = np.zeros(shape=(50000,100))
        data_images = np.zeros(shape=(50000,64,64,3))
        len_files = len(files)
        for i in range(len(files)):
            class_path = train_dir + '/' + files[i] + '/images/'
            images = os.listdir(class_path)
            len_images = len(images)
            for j in range(len(images)):
                img = cv2.imread(class_path+images[j])
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_mat = np.array(image)
                index = i*len_images + j
                data_images[index] = np.array(image_mat)
                data_labels[index, i] = 1

        data_images = data_images.reshape((50000, 3*64*64))

        print np.shape(data_labels)
        print np.shape(data_images)
        #print data_labels[5005]

        return data_images,data_labels
def read_validation_set():
    files = os.listdir(validation_dir)
    class_paths = []
    data_label = np.zeros(shape=(5000,100))
    data_image = np.zeros(shape=(5000,64,64,3))
    #print len(files)
    len_files = len(files)
    for i in range(len(files)):
        class_path = validation_dir + '/' + files[i] + '/images/'
        images = os.listdir(class_path)
        len_images = len(images)
        #print len_images
        for j in range(len(images)):
            img = cv2.imread(class_path+images[j])
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_mat = np.array(image)
            index = i*len_images + j
            #print index
            data_image[index] = np.array(image_mat)
            data_label[index, i] = 1

    data_image = data_image.reshape((5000, 3*64*64))

    print np.shape(data_label)
    print np.shape(data_image)

    print data_label[0]

    return data_image,data_label

def rnn_neural_network_model(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
                 'biases':tf.Variable(tf.random_normal([n_classes]))}
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, n_chunks,x)

#    lstm = rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=False)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    stacked_lstm = rnn_cell.MultiRNNCell([lstm_cell] * number_of_layers)
    outputs, states = rnn.rnn(stacked_lstm, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
    train_images, train_labels = read_training_set()
    test_images, test_labels = read_validation_set()

    prediction = rnn_neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    num_examples = 50000
    config = tf.ConfigProto(allow_soft_placement = True, device_count = {'GPU': 1})
    with tf.Session(config = config) as sess:
        sess.run(tf.initialize_all_variables())

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(int(num_examples/batch_size)):
                s_i = i * batch_size
                e_i = s_i + batch_size
                epoch_x, epoch_y = train_images[s_i:e_i], train_labels[s_i:e_i]
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

                i, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Accuracy:',accuracy.eval({x:test_images.reshape((-1,n_chunks, chunk_size)), y:test_labels}))
        feed_dict = {x: epoch_x}
        classification = sess.run(feed_dict)

        print np.shape(test_images)
        print np.shape(test_labels)


def main():
    train_neural_network(x)

if __name__=='__main__':
    main()
