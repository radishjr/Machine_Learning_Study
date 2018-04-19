
#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import iris_data
import os

import numpy as np
import pandas as pd
import random
from tensorflow.contrib.tensorboard.plugins import projector

current_path = os.path.dirname(os.path.abspath(__file__)) + "/"

tf.set_random_seed(1)

epch = 1000
batch_size = 100
LR = 1
regularize = True
use_layer = True

summaryPath = current_path + 'training/train_lr%g_batch%d_regularize%d_layer%d'%(LR,batch_size,regularize,use_layer)
saverPath = summaryPath + "/model.ckpt"

def GetNN():
    #Generate Nuraul Network

    feature_size = 4
    category_size = 3
    layer_1_size = 10
    layer_2_size = 10

    real_X = tf.placeholder(tf.float32, [None, feature_size], name="X")          
    real_Y = tf.placeholder(tf.float32, [None, category_size], name="Label")

    with tf.variable_scope('Hidden_Layer_1'):
        if use_layer:
            layer_1 = tf.layers.dense(real_X, layer_1_size, activation=tf.nn.relu)        
        else:
            weight_1 = tf.Variable(tf.random_normal([feature_size, layer_1_size]), name="W1")
            bias_1 = tf.Variable(tf.zeros([layer_1_size]), name="B1")
            matmul_output_1 = tf.matmul(real_X, weight_1) + bias_1
            layer_1 = tf.nn.relu(matmul_output_1)

    with tf.variable_scope('Hidden_Layer_2'):
        if use_layer:
            layer_2 = tf.layers.dense(layer_1, layer_2_size, activation=tf.nn.relu)
        else:
            weight_2 = tf.Variable(tf.truncated_normal([layer_1_size, layer_2_size]), name="W2")
            bias_2 = tf.Variable(tf.zeros([layer_2_size]), name="B2")
            matmul_output_2 = tf.matmul(layer_1, weight_2) + bias_2
            layer_2 = tf.nn.relu(matmul_output_2)

    with tf.variable_scope('Output_Layer'):
        if use_layer:
            predict = tf.layers.dense(layer_2, category_size, activation=tf.nn.softmax)
        else:
            weight_3 = tf.Variable(tf.truncated_normal([layer_2_size, category_size]), name="W3")
            bias_3 = tf.Variable(tf.zeros([category_size]), name="B3")
            matmul_output_3 = tf.matmul(layer_2, weight_3) + bias_3
            predict = tf.nn.softmax(matmul_output_3)
   
    #loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=real_Y, logits=predict)
    #loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.argmax(real_Y, 1), logits=predict)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=real_Y, logits=predict)

    train = tf.train.GradientDescentOptimizer(LR).minimize(loss)
    #train = tf.train.AdamOptimizer(LR).minimize(loss)
    #train = tf.train.ProximalAdagradOptimizer(LR).minimize(loss)

    correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(real_Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    
    return real_X, real_Y, predict, loss, train, accuracy

def generateData():
    # Fetch the data
    (train_x, train_y), (test_x, test_y) = iris_data.load_data_onehot()
    predict_x = pd.DataFrame({
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    })

    if regularize:
        allData = pd.concat([train_x, test_x, predict_x])
        allData_norm = (allData - allData.mean()) / (allData.max() - allData.min())

        train_x = allData_norm[:train_x.shape[0]]
        test_x = allData_norm[train_x.shape[0]:train_x.shape[0]+test_x.shape[0]]
        
        predict_x = allData_norm[train_x.shape[0]+test_x.shape[0]:]

    return (train_x, train_y), (test_x, test_y), (predict_x)


real_X, real_Y, predict, loss, train, accuracy = GetNN()


(train_x, train_y), (test_x, test_y), (predict_x) = generateData()


merged = tf.summary.merge_all()

saver = tf.train.Saver()

with tf.Session() as sess:    
    
    #################### TRAIN ####################
    
    summary_writer = tf.summary.FileWriter(summaryPath, sess.graph)   
    sess.run(tf.global_variables_initializer())

    if os.path.exists(saverPath) or os.path.exists(saverPath+".index"):
        saver.restore(sess, saverPath)

    for i in range(epch):

        X_index = random.sample(train_x.index.tolist(), batch_size)
        train_X_batch = train_x.ix[X_index]
        train_y_batch = train_y.ix[X_index]

        summary, _ =sess.run([merged, train], feed_dict={
            real_X: train_X_batch, 
            real_Y: train_y_batch
        })

        if i % 50 == 0:
            summary_writer.add_summary(summary, i)
            
            l = sess.run(loss, feed_dict={
                real_X: train_x, 
                real_Y: train_y
            })
            acc = sess.run(accuracy, feed_dict={
                real_X: train_x, 
                real_Y: train_y
            })

            print("%d: loss:%f, accuracy:%f"%(i, l,acc))    

            #Save variables
            save_path = saver.save(sess, saverPath)
    
    summary_writer.flush()
        
    #################### PREDICT ####################

    pred_new = sess.run(predict, feed_dict={
        real_X: predict_x
    })

    pred_new_df = pd.DataFrame(pred_new)
    
    expected = [0, 1, 2]

    predicted = pred_new_df.idxmax(axis=1)
    predicted_pct = pred_new_df.max(axis=1)
    print("pred_new: {}".format(pred_new))
    print("predicted: {}".format(predicted))
    for i in range(predicted.shape[0]):
        print("pretict %d (accuracy: %f), expected: %d" %(predicted[i], predicted_pct[i], expected[i]))
