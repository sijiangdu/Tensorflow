# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2017 Sijiang Du
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Credit Card Fraud Detection
Multi-layer neoral network is trained by the data set ofaAnonymized credit card transactions labeled as fraudulent or genuine
There are two classes of labeled rows: fraud and normal(true) transactions The total fraud trainsactions are less than 0.2%
among overall 284807 records. The program constructs a neural network model to detect the fraud transactions. 
The training has several epochs. Each training step is done in a batch inputs. Half of the batch data are fraud data and another half
are normal transactions data.
The data set is divided to two parts: 80% for training set and 20% for testing set.
The testing result is evaluated by two measurements: overall accuracy and KS value,
The Kolmogorov Smirnov chart is ploted as result and a PNG file is saved in FLAGS.result_dir folder.

Tensorflow input_pipeline is being used to streamline and randomlize inputs for the training and testing session.
To start: call the train() function, e.g. "train(5,100)", means 5 hidden layers, each layer has 100 neurons.  

Author: Sijiang Du, Updated in May 2, 2017

This file is a modified file of https://www.github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py
In this file:
The input data are in "creditcard.csv" which is downloaded from https://www.kaggle.com/dalpozz/creditcardfraud
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

import numpy as np
import os.path
import csv
import random

csv_file_name = "creditcard.csv"
train_file_name_true = "creditcard_train_true.csv"
train_file_name_fraud = "creditcard_train_fraud.csv"
test_file_name_true = "creditcard_test_true.csv"
test_file_name_fraud = "creditcard_test_fraud.csv"
test_file_name_both =  "creditcard_test_all.csv"

batch_size = 128
input_channel_num = 30
output_channel_num = 2
hidden_layer_neuron = 200
hidden_layer_num = 3
train_epoch = 3
csv_lines_train = 0
csv_lines_test_both = 0
 
# partition: randomly partion the input file to two files. E.g. partition = 0.8: 80% for training 20% for testing.
def csv_partition_train_test(input_file, partition=0.8):
  with open(input_file) as data:
      with open(FLAGS.data_dir+test_file_name_true, 'w+') as test_true,\
           open(FLAGS.data_dir+test_file_name_fraud, 'w+') as test_fraud,\
           open(FLAGS.data_dir+test_file_name_both, 'w+') as test_both:
          with open(FLAGS.data_dir+train_file_name_true, 'w+') as train_true, open(FLAGS.data_dir+train_file_name_fraud, 'w+') as train_fraud:
              header = next(data)
 #             test.write(header)
 #             train.write(header)           
              csv_r = csv.reader(data)
              csv_w_train_true = csv.writer(train_true)
              csv_w_train_fraud = csv.writer(train_fraud)
              csv_w_test_true = csv.writer(test_true)
              csv_w_test_fraud = csv.writer(test_fraud)
              csv_w_test_both = csv.writer(test_both)             
              global csv_lines_test_both
              global csv_lines_train
              
              for line in csv_r:
                
                  if line[-1]=='0':
                      line = line[:-1] + ['0','1']  
                  elif line[-1]=='1':
                      line = line[:-1] + ['1','0']
                       
                  if random.random() < partition:
                    csv_lines_train +=1
                    if line[-1]=='0':
                       csv_w_train_fraud.writerow(line)
                    else:
                       csv_w_train_true.writerow(line)
                       
                  else:
                    csv_w_test_both.writerow(line)
                    csv_lines_test_both += 1
                    if line[-1]=='0':
                       csv_w_test_fraud.writerow(line)
                    else:
                       csv_w_test_true.writerow(line)


def read_creditcard_csv(filename_queue):

  reader = tf.TextLineReader(skip_header_lines=1)
  key, value = reader.read(filename_queue)
  record_defaults = [[0.0]for row in range(32)]
  cols = tf.decode_csv(
  value, record_defaults=record_defaults)
  features = tf.stack(cols[:-2])
  label = tf.stack([cols[30],cols[31]])
  return features, label

def input_pipeline(filenames, batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example, label = read_creditcard_csv(filename_queue)
  # min_after_dequeue defines how big a buffer we will randomly sample
  #   from -- bigger means better shuffling but slower start up and more
  #   memory used.
  # capacity must be larger than min_after_dequeue and the amount larger
  #   determines the maximum we will prefetch.  Recommendation:
  #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example_batch, label_batch

def train(layer_num= hidden_layer_num, neuron_num = hidden_layer_neuron):

  if    not os.path.exists(FLAGS.data_dir+train_file_name_true)\
     or not os.path.exists(FLAGS.data_dir+train_file_name_fraud)\
     or not os.path.exists(FLAGS.data_dir+test_file_name_true)\
     or not os.path.exists(FLAGS.data_dir+test_file_name_fraud):                    
    csv_partition_train_test(csv_file_name, 0.8)
 
  sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, input_channel_num], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, output_channel_num], name='y-input')

  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, keep_prob=1.0):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        print(weights)
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      if FLAGS.dropout<1.0: dropped = tf.nn.dropout(activations, keep_prob)
      return activations
    
  keep_prob = tf.placeholder(tf.float32)
  
  #Add layer_num layers of hidden layer
  cur_layer = nn_layer(x, input_channel_num, neuron_num, 'layer_1', tf.identity, keep_prob)
  for i in range(1,layer_num):
    cur_layer = nn_layer(cur_layer, neuron_num, neuron_num, 'layer_'+str(i), tf.identity, keep_prob)

  # the last layer is the output layer
  y = nn_layer(cur_layer, neuron_num, output_channel_num, 'output_layer', act=tf.identity)
  # scale the activation down by the size of hidden layer
  y = y/neuron_num

  with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  #scorecard for the model evaluation. A lower score indictates a higher probability to be fraud.
  with tf.name_scope('score_card'):
    yc = tf.reshape( y[:,1]-y[:,0], (-1,1))
    yc_ = tf.reshape( y_[:,1]-y_[:,0], (-1,1))
    score_card = tf.concat([yc, yc_], 1)
  
  with tf.name_scope('input_examples'):
      example_batch_train_true, label_batch_train_true = input_pipeline(tf.constant([FLAGS.data_dir+train_file_name_true]), round(batch_size/2))
      example_batch_train_fraud, label_batch_train_fraud = input_pipeline(tf.constant([FLAGS.data_dir+train_file_name_fraud]), batch_size-round(batch_size/2))
      example_batch_test_true, label_batch_test_true = input_pipeline(tf.constant([FLAGS.data_dir+test_file_name_true]), batch_size)
      example_batch_test_fraud, label_batch_test_fraud = input_pipeline(tf.constant([FLAGS.data_dir+test_file_name_fraud]), batch_size)
      example_batch_test_both, label_batch_test_both = input_pipeline(tf.constant([FLAGS.data_dir+test_file_name_both]), batch_size,1)
      
  init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

  # Merge all the summaries and write them out to /tmp/tensorflow/creditcard/logs/creditcard_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer_true = tf.summary.FileWriter(FLAGS.log_dir + '/test_true')
  test_writer_fraud = tf.summary.FileWriter(FLAGS.log_dir + '/test_fraud')

  sess.run(init_op)
  # Start input enqueue threads.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train,test=0):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
      batch_xs_1, batch_ys_1 = sess.run([example_batch_train_true, label_batch_train_true])
      batch_xs_2, batch_ys_2 = sess.run([example_batch_train_fraud, label_batch_train_fraud])

      xs = np.concatenate ((batch_xs_1,batch_xs_2))
      ys = np.concatenate ((batch_ys_1,batch_ys_2))
      perm = np.random.permutation(xs.shape[0])
      np.take(xs,perm,axis=0,out=xs)
      np.take(ys,perm,axis=0,out=ys)
      k = FLAGS.dropout
    else:
      if test==0:
        xs, ys = sess.run([example_batch_test_fraud, label_batch_test_fraud])
      elif test==1:
        xs, ys = sess.run([example_batch_test_true, label_batch_test_true])
      elif test==2:
        xs, ys = sess.run([example_batch_test_both, label_batch_test_both])
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  scorecard_all = [[0, 0]]
  acc_all = []
  print('[Training inputs: %s, epoch = %s], [Testing inputs: %s],' % (csv_lines_train, train_epoch,csv_lines_test_both))
  try:      
      i = 0
      train_s = csv_lines_train*train_epoch/batch_size
      # plotting
      fig1 = plt.figure(1)
      ax = fig1.add_subplot(111)
      plt.ylim((-0.5, 1.5))
      plt.ylabel('Accuracy')
      plt.xlabel('Training Batches')
      ax.text(0.05, 0.90,'Fraud: (red) --', ha='left', va='center', color='red', transform=ax.transAxes)
      ax.text(0.05, 0.85,'Normal: (blue) --', ha='left', va='center', color='blue', transform=ax.transAxes)
      fig1.suptitle('Tensorflow: Creditcard Fraud Detection Training')
      while not coord.should_stop():
            i = i+1

            if i<=train_s :
               #Record train set summaries, and train
              if i % 100 == 0:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _= sess.run([merged, train_step],
                                      feed_dict=feed_dict(True),
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
                
              else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)

              if i % 10 == 0:  # Record summaries and test-set accuracy
                  summary, acc0 = sess.run([merged, accuracy], feed_dict=feed_dict(False,0))
                  test_writer_fraud.add_summary(summary, i)
                  summary, acc1 = sess.run([merged, accuracy], feed_dict=feed_dict(False,1))
                  test_writer_true.add_summary(summary, i)
                  print('%s: [%s], [%s]' % (i, acc0,acc1))
                  
                  # plotting
                  plt.plot(list(range(i-10,i)),[acc0]*10, 'r', list(range(i-10,i)),[acc1]*10, 'b')
                  if i%100 == 0:
                    plt.draw()
                    plt.pause(0.3)

            else:
              acc, scorecard = sess.run([accuracy,score_card], feed_dict=feed_dict(False,2))
              acc_all.append(acc)
              scorecard_all = np.concatenate ((scorecard_all,scorecard))              

  except tf.errors.OutOfRangeError:
      print('Done training and testing -- epoch limit reached')
  finally:
      # When done, ask the threads to stop.
      coord.request_stop()

  scorecard_all = np.delete(scorecard_all, (0), axis=0)
  print(scorecard_all.shape)
  scorecard_all = scorecard_all[scorecard_all[:, 0].argsort()]
  count_true = 0
  count_fraud = 0
  percent_true = np.zeros(scorecard_all.shape[0])
  percent_fraud = np.zeros(scorecard_all.shape[0])
  score_range = []
  score_std = scorecard_all[:,0].std()
  
  for i in range(scorecard_all.shape[0]):
      
      if scorecard_all[i,1] > 0:
          count_true +=1
      else: 
          count_fraud +=1
      percent_true[i] = count_true
      percent_fraud[i] = count_fraud
  ks = 0
  ks_pos = 0
  for i in range(scorecard_all.shape[0]):
      percent_true[i]  /= count_true
      percent_fraud[i] /= count_fraud
      diff = percent_fraud[i] - percent_true[i]
      if diff > ks:
        ks = diff
        ks_pos = i
  accuracy_mean = np.mean(acc_all)
  print('Scorecard standard deviation:', score_std)
  acc = 'Accuracy (' + str(csv_lines_test_both) + ' samples): ' + str(accuracy_mean)
  print(acc)
  fig2 = plt.figure(2,figsize=(10, 8))
  ax = fig2.add_subplot(111)
  plt.plot(list(scorecard_all[:, 0]),percent_fraud, 'r', list(scorecard_all[:, 0]),percent_true, 'b')
  plt.plot([scorecard_all[ks_pos,0],scorecard_all[ks_pos,0]], [percent_true[ks_pos], percent_fraud[ks_pos]], 'g')
  ax.text(0.05, 0.95,'Greenline: KS = '+ str(ks), ha='left', va='center', color='green', transform=ax.transAxes)
  ax.text(0.05, 0.90,'Redline: Fraud', ha='left', va='center', color='red', transform=ax.transAxes)
  ax.text(0.05, 0.85,'Blueline: Normal', ha='left', va='center', color='blue', transform=ax.transAxes)
  ax.text(0.05, 0.80,acc, ha='left', va='center', color='black', transform=ax.transAxes)
  plt.ylabel('Cumulative percentage')
  plt.xlabel('Scorecard scoring')
  fig2.suptitle('KS graph (Creditcard Fraud Detection by Tensorflow)\n('+str(layer_num) +' hidden layers, '+str(neuron_num)+' neurons per layer)')
  result_str = str(round(int(accuracy_mean*1000)))+'_'+str(int(round(ks*1000)))+'_'+str(layer_num)+'_'+str(neuron_num)+'_e'+str(train_epoch)
  plt.draw()
  plt.savefig(FLAGS.result_dir+'/creditcard_ks_'+result_str+'.png')
  plt.pause(3) 
  # Wait for threads to finish.
  coord.join(threads)

  # Wait for threads to finish.
  coord.join(threads)

  sess.close()

  train_writer.close()
  test_writer_true.close()
  test_writer_fraud.close()
  return ks,accuracy_mean

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  if tf.gfile.Exists(FLAGS.data_dir):
    tf.gfile.DeleteRecursively(FLAGS.data_dir)
  tf.gfile.MakeDirs(FLAGS.data_dir)
  if not tf.gfile.Exists(FLAGS.result_dir):
    tf.gfile.MakeDirs(FLAGS.result_dir)
 # 4 hidden layers, each layer has 20 output channels. One layer is a (20,20) tensor.
  train(4, 20)
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default= 1.0,
                      help='Keep probability for training dropout.')
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/creditcard/input_data/',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/creditcard/logs/creditcard_with_summaries',
                      help='Summaries log directory')
  parser.add_argument('--result_dir', type=str, default='/tmp/tensorflow/creditcard/result',
                      help='Summaries log directory')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
