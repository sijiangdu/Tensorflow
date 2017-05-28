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
# Author: Sijiang Du, May, 2017
# This program runs on MAC OS. It is for Kaggle Digit Recognizer competition.
#  
# Input: train.csv, test.csv. 
# 
# Output: "submission_sijiangdu.csv" in folder /tmp/tensorflow/mnist_rnn/input_data/
# Generated temparary files are at /tmp/tensorflow/mnist_rnn/
#
# Kaggle.com provides its own MNIST "train.csv" and "test.csv".
# The program here uses tensorflow input pipe line to stream line the training and testing data from the cvs files.
# There are two flags in the program "train_by_mnist_lib" and "test_by_mnist_lib" set those to "True" to use moist_data as inputs.
# Otherwise, user need to download the cvs file from Kaggle website for Digit Recognizer: https://www.kaggle.com/c/digit-recognizer/data 
#
# A 2-D convolution network is implemented.
# The variable "n_conv" defines the number of convolution layer being added. The 2-D pooling is applied after 2-D covn.
# However, there is no pooling if the image width is too mall (width is less than 10).
#
# The program provided an emxaple how to construct a CNN with a sigle parameter and add multiple conv layers accordingly in the while loop.
# Two-layer "n_conv = 2" is the default value. The vaiable value can be changed to larger number to construct a much deeper network in the
# CNN_Wrapper function.
# 

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import sys
import numpy as np
import os.path
import csv
import random

csv_file_name = "train.csv"
test_csv_file_name = "test.csv"
train_file_name = "mnist_rnn_train.csv"
test_file_name = "mnist_rnn_test.csv"
submit_test_file_name = "test_28000.csv"
submit_result_file_name = "submission_sijiangdu.csv"

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

## this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
csv_size = 42000
train_size = 40000
test_size = 2000
epochs = 5
training_iters = train_size * epochs
batch_size = 100
img_width = 28
img_height = 28
sample_size = img_width*img_height
n_conv = 6

n_classes = 10      # MNIST classes (0-9 digits)
  
# partition: randomly partion the input file to two files. E.g. partition = 0.8: 80% for training 20% for testing.
def csv_partition_train_test(input_file, partition=0.98):
  global csv_size, train_size, test_size,training_iters
  csv_size = 0
  train_size = 0
  test_size = 0
  with open(input_file) as data:
      with open(FLAGS.data_dir+test_file_name, 'w+') as test:
          with open(FLAGS.data_dir+train_file_name, 'w+') as train:
              header = next(data)
              test.write(header)
              train.write(header)           
              csv_r = csv.reader(data)
              csv_w_train = csv.writer(train)
              csv_w_test = csv.writer(test)
              for line in csv_r:
                  csv_size += 1
                  if(len(line)!=785):
                    print("Invalid CSV format. Discard record #%s"%(csv_size))
                    continue
                   
                  if random.random() < partition:
                    csv_w_train.writerow(line)
                    train_size += 1
                  else:
                    csv_w_test.writerow(line)
                    test_size += 1
                    
  training_iters = train_size * epochs
  print("CSV input size =  %s, train set size = %s, validation set size = %s, training samples = %s"%(csv_size , train_size,test_size, training_iters))
  
#add a dummy column to the test.csv
def csv_test_csv_file_change(input_file, output_file):
  with open(input_file) as data:
      with open(FLAGS.data_dir+output_file, 'w+') as out_file:
              header = next(data)
              out_file.write("label,"+header)
              csv_r = csv.reader(data)
              csv_w = csv.writer(out_file)
              size = 0
              for line in csv_r:
                  size += 1
                  line = [-1] + line
                  if(len(line)!=785):
                    print("Invalid test.csv. Discard record #%s"%(size))
                    continue

                  csv_w.writerow(line)
  print("test.csv input size =  %s"%(size))
                      
def read_mnist_csv(filename_queue):
  reader = tf.TextLineReader(skip_header_lines=1)
  key, value = reader.read(filename_queue)
  record_defaults = [[0]for row in range(785)]
  cols = tf.decode_csv( value, record_defaults=record_defaults)
  features = tf.stack(cols[1:])
  label = tf.stack([cols[0]])
  return features, label

def input_pipeline(filenames, batch_size, num_epochs=None, shuffle=True):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=shuffle)
  features, label = read_mnist_csv(filename_queue)
  # min_after_dequeue defines how big a buffer we will randomly sample
  #   from -- bigger means better shuffling but slower start up and more
  #   memory used.
  # capacity must be larger than min_after_dequeue and the amount larger
  #   determines the maximum we will prefetch.  Recommendation:
  #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  if shuffle == True:
      feature_batch, label_batch = tf.train.shuffle_batch(
          [features, label], batch_size=batch_size, capacity=capacity,
          min_after_dequeue=min_after_dequeue)
  else:
      feature_batch, label_batch = tf.train.batch(
          [features, label], batch_size=batch_size, capacity=capacity
          )
  return feature_batch, label_batch

# display image
show_num = 5
fig_mnist, ax_array = plt.subplots(show_num,show_num)

def show_mnist(images,labels,title = "Digits"):
  global fig_mnist, ax_array
  plt.figure(fig_mnist.number)
  fig_mnist.suptitle(title)
  n = len(images)
  z = np.zeros((28,28))
  t = [[i] for i in range(10)]
  for i in range(show_num*show_num):
    row = int(i/show_num)
    col = int(i%show_num)

    if i<n:
      img = images[i].reshape(28,28)
      ax_array[row,col].imshow(img, cmap=cm.binary)
      ax_array[row, col].set_title(int(labels[i]))
      ax_array[row, col].axis('off')
    else:
      ax_array[row, col].imshow(z, cmap=cm.binary)
      ax_array[row, col].set_title('')
      ax_array[row, col].axis('off')
  plt.draw()
  plt.pause(0.3)
  plt.savefig(FLAGS.result_dir+'/'+ title +'.png')

# tf Graph input
x = tf.placeholder(tf.float32, [None, 28*28])   
y = tf.placeholder(tf.float32, [None, n_classes])


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

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def CNN_Wrapper(X, num_classes, n_hidden_layers, init_size=32, drop_out = 0.9, name='Conv_2D'):
  with tf.name_scope(name):
    x = tf.reshape(X, [-1, 28, 28, 1])
    print( "CNN input:"+str(x) )
    width = 28;
    #Add layer_num layers of hidden layer
    W = weight_variable([5,5, 1,init_size])
    b = bias_variable([init_size])
    cur_layer = conv2d(x, W) + b
    cur_layer = tf.nn.relu(cur_layer)
    cur_layer = max_pool_2x2(cur_layer)
    width /= 2;
    size = init_size
    print("CNN layer 1: " + str(cur_layer) )
    for i in range(1,n_hidden_layers):
      W = weight_variable([5,5,size,size*2])
      b = bias_variable([size*2])
      cur_layer = conv2d(cur_layer, W) + b
      cur_layer = tf.nn.relu(cur_layer)
      if width > 7:  #not doing pooling when the width is small
        cur_layer = max_pool_2x2(cur_layer)
        width = (int)(width/2);
      size = size*2
      print('CNN layer %s: '%(i+1) + str(cur_layer) )

    height = width
    W = weight_variable([width*height*size, 1024])
    b = bias_variable([1024])
    #flat the layer: [n_samples, 7, 7, size] ->> [n_samples, 7*7*size]
    cur_layer = tf.reshape(cur_layer, [-1, width*height*size])
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, W) + b)
    cur_layer = tf.nn.dropout(cur_layer, drop_out)
    print('CNN output flat layer:'+str(cur_layer))

    # output results#
    W = weight_variable([1024, num_classes])
    b = bias_variable([num_classes])
    results = tf.matmul(cur_layer, W) + b
    results = tf.nn.l2_normalize(results,0)
    return results
 
  
def train():

    if   not os.path.exists(FLAGS.data_dir+train_file_name)\
      or not os.path.exists(FLAGS.data_dir+test_file_name):
      csv_partition_train_test(csv_file_name)
    if not os.path.exists(FLAGS.data_dir+submit_test_file_name):
      csv_test_csv_file_change(test_csv_file_name, submit_test_file_name)
      
    pred  = CNN_Wrapper(x, n_classes, n_conv, init_size=32, drop_out = FLAGS.dropout, name='Conv_2_layer');

    with tf.name_scope('Train'):
      cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
      train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cost)

    with tf.name_scope('Classify'):
      classification = tf.argmax(pred, 1)
      with tf.name_scope('accuracy'):
          with tf.name_scope('correct_prediction'):
            correct_pred = tf.equal(classification, tf.argmax(y, 1))
          accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
          tf.summary.scalar('accuracy', accuracy)
    
    with tf.name_scope('Input_Batch'):
      example_batch_train, label_batch_train = input_pipeline(tf.constant([FLAGS.data_dir+train_file_name]), batch_size)
      example_batch_test, label_batch_test = input_pipeline(tf.constant([FLAGS.data_dir+test_file_name]), batch_size)
      example_batch_submit, label_batch_submit = input_pipeline(tf.constant([FLAGS.data_dir+submit_test_file_name]), batch_size,num_epochs=1,shuffle=False)

    train_by_mnist_lib = False
    test_by_mnist_lib = True
    def feed_dict(train, submit=False):
      """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
      if train:
        if train_by_mnist_lib == True:
            xs, ys = mnist.train.next_batch(batch_size)
            return {x: xs, y: ys}
          
        xs, ys_label = sess.run([example_batch_train, label_batch_train])
      else:
        if submit:
          xs, ys_label = sess.run([example_batch_submit, label_batch_submit])
        else:
          if test_by_mnist_lib == True:
            xs, ys = mnist.test.next_batch(batch_size)
            return {x: xs, y: ys}
          
          xs, ys_label = sess.run([example_batch_test, label_batch_test])
           
      n = ys_label.shape[0]
      ys = np.zeros((n,10))
      if not submit:
        for i in range(n):
          ys[i][int(ys_label[i])] = 1
      xs = xs/255    
      return {x: xs, y: ys}

    sess = tf.InteractiveSession()
    with tf.name_scope('training_epoch'):
        # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 0
        acc = 0
        # plotting
        fig1, ax = plt.subplots(1,1)
        plt.ylim((0.94, 1.03))
        plt.ylabel('Accuracy')
        plt.xlabel('Training Step ( bactch size:' + str(int(batch_size)) + ')')
        ax.text(0.05, 0.90,'Number of Conv layers: ' + str(n_conv), ha='left', va='center', color='blue', transform=ax.transAxes)
        text_acc = ax.text(0.05, 0.85,'Accuracy: ' + str(acc), ha='left', va='center', color='green', transform=ax.transAxes)
        fig1.suptitle('Tensorflow CNN - MNIST Digit Recognizer')
        acc_all = [0.0,0.0,0.0]
        plt.draw()
        plt.pause(0.3)
        
        try:
          submit_file = open(FLAGS.data_dir+submit_result_file_name, 'w+')
          submit_file.write("ImageId,Label\r\n")
          csv_w_submit = csv.writer(submit_file)
          submit_size = 0

          while not coord.should_stop():

              #generate output submission file
              if step * batch_size > training_iters:
                feed_d = feed_dict(False,True)
                [digits] = sess.run([classification], feed_d)
                for i in digits:
                    submit_size += 1
                    line = [str(submit_size),str(i)]
                    csv_w_submit.writerow(line)
               
                if(submit_size%1000 == 0):
                  print("Outputs to submission file: "+str(submit_size))
                continue                 

              #training
              sess.run([train_op], feed_dict=feed_dict(True))
              step += 1

              #testing and plotting training progress
              test_at = 100
              if step % test_at == 0:
                  tmp = acc
                  #Init images those are incorrectly classified
                  s = np.zeros((1,28*28))
                  d = np.zeros(1)
                  test_loop = 100
                  acc_l = [0.0]*test_loop
                  for i in range(test_loop):
                      feed_d=feed_dict(False)
                      acc_l[i],summary,digits = sess.run([accuracy,merged,classification], feed_dict=feed_d)
                      train_writer.add_summary(summary, step)
                      
                      #show the images those are incorrectly classified
                      if len(s) > show_num*show_num: continue
                      correct = np.argmax(feed_d[y],1)
                      for i in range(batch_size):
                        if correct[i] != digits[i]:
                          s = np.append(s,np.array([feed_d[x][i].flatten()]),0)
                          d = np.append(d,np.array([digits[i]]),0)
                  acc = np.mean(acc_l)
                  show_mnist(s[1:],d[1:],"Incorect Classifications")
                  print(acc)
                  plt.figure(fig1.number)
                  plt.plot([step-test_at,step], [tmp, acc],'g')
                  acc_all.append(acc)
                  acc_all.pop(0)
                  text_acc.set_text('Accuracy:  ..., [%s], [%s], [%s]'%(acc_all[-3], acc_all[-2],acc_all[-1]) )    
                  plt.draw()
                  plt.savefig(FLAGS.result_dir+'/mnist_cnn'+'.png')
                  plt.pause(0.3)
                
        except tf.errors.OutOfRangeError:
          print('Done training and testing -- epoch limit reached')
        finally:
          # When done, ask the threads to stop.
          coord.request_stop()
          
        submit_file.close()  
        if not coord.should_stop():
          coord.request_stop()
        # Wait for threads to finish.
        coord.join(threads)
        sess.close()

        result_str = str(round(int(acc*1000)))+'_layer_'+str(n_conv)
        plt.figure(fig1.number)
        plt.savefig(FLAGS.result_dir+'/mnist_cnn_'+result_str+'.png')
        train_writer.close()

def main(_):
    
    if tf.gfile.Exists(FLAGS.log_dir):
      tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    if tf.gfile.Exists(FLAGS.data_dir):
      tf.gfile.DeleteRecursively(FLAGS.data_dir)
    if not tf.gfile.Exists(FLAGS.data_dir):
      tf.gfile.MakeDirs(FLAGS.data_dir)
    if not tf.gfile.Exists(FLAGS.result_dir):
      tf.gfile.MakeDirs(FLAGS.result_dir)
    #enter the training and testing loop     
    train()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default= 0.95,
                      help='Keep probability for training dropout.')
  parser.add_argument('--forget_bias', type=float, default= 0.9,
                      help='forget bias for training')
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist_rnn/input_data/',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist_rnn/logs/mnist_rnn_with_summaries',
                      help='Summaries log directory')
  parser.add_argument('--result_dir', type=str, default='/tmp/tensorflow/mnist_rnn/result',
                      help='result plotting PNG files directory')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    
