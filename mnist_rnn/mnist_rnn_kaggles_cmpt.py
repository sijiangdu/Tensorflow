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
epochs = 8
training_iters = train_size * epochs
batch_size = 100
sample_size = 28*28
n_cell= 400
n_steps = 28    #4 # time steps
n_classes = 10      # MNIST classes (0-9 digits)
 

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, 28*28/n_steps])   
y = tf.placeholder(tf.float32, [None, n_classes])


def RNN_Wrapper(X, num_classes, n_hidden_units, forget_bias = 1.0,  name='Basic_LSTM'):

  with tf.name_scope(name):
    W = tf.Variable(tf.truncated_normal([n_hidden_units, num_classes],stddev=0.1))
    b = tf.Variable(tf.zeros([num_classes]))
  
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias = forget_bias)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell, X,sequence_length=[sample_size for i in range(batch_size)], initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], W) + b     
    return results
  
# partition: randomly partion the input file to two files. E.g. partition = 0.8: 80% for training 20% for testing.
def csv_partition_train_test(input_file, partition=0.952):
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
  print("CSV input size =  %s, train set size = %s, validation set size = %s"%(csv_size , train_size,test_size))
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

def input_pipeline(filenames, batch_size, num_epochs=None):
  print(filenames)
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  features, label = read_mnist_csv(filename_queue)
  # min_after_dequeue defines how big a buffer we will randomly sample
  #   from -- bigger means better shuffling but slower start up and more
  #   memory used.
  # capacity must be larger than min_after_dequeue and the amount larger
  #   determines the maximum we will prefetch.  Recommendation:
  #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  feature_batch, label_batch = tf.train.shuffle_batch(
      [features, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return feature_batch, label_batch


# display image
show_num = 3
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
    
def train():

    if   not os.path.exists(FLAGS.data_dir+train_file_name)\
      or not os.path.exists(FLAGS.data_dir+test_file_name):
      csv_partition_train_test(csv_file_name)
    if not os.path.exists(FLAGS.data_dir+submit_test_file_name):
      csv_test_csv_file_change(test_csv_file_name, submit_test_file_name)
      
    pred = RNN_Wrapper(x, n_classes, n_cell, FLAGS.forget_bias)
    with tf.name_scope('train'):
      cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
      train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cost)

    with tf.name_scope('accuracy'):
      with tf.name_scope('correct_prediction'):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
      with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
      with tf.name_scope('classification'):
        classification = tf.argmax(pred, 1)
        
    tf.summary.scalar('accuracy', accuracy)
    
    with tf.name_scope('input_images'):
      example_batch_train, label_batch_train = input_pipeline(tf.constant([FLAGS.data_dir+train_file_name]), batch_size)
      example_batch_test, label_batch_test = input_pipeline(tf.constant([FLAGS.data_dir+test_file_name]), batch_size)
      example_batch_submit, label_batch_submit = input_pipeline(tf.constant([FLAGS.data_dir+submit_test_file_name]), batch_size,1)

    test_by_mnist_lib = False
    train_by_mnist_lib = True
    def feed_dict(train, submit=False):
      """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
      if train:
        if train_by_mnist_lib == True:
            xs, ys = mnist.train.next_batch(batch_size)
            xs = xs.reshape([batch_size, n_steps, -1])
            return {x: xs, y: ys}
          
        xs, ys_label = sess.run([example_batch_train, label_batch_train])
      else:
        if submit:
          xs, ys_label = sess.run([example_batch_submit, label_batch_submit])
        else:
          if test_by_mnist_lib == True:
            xs, ys = mnist.test.next_batch(batch_size)
            xs = xs.reshape([batch_size, n_steps, -1])
            return {x: xs, y: ys}
          
          xs, ys_label = sess.run([example_batch_test, label_batch_test])
           
      xs = xs.reshape([batch_size, n_steps, -1])
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
        plt.ylim((-0.5, 1.5))
        plt.ylabel('Accuracy')
        plt.xlabel('Training Step ( bactch size:' + str(int(batch_size)) + ')')
        ax.text(0.05, 0.90,'Max Time -Steps in RNN cell: ' + str(n_steps), ha='left', va='center', color='blue', transform=ax.transAxes)
        ax.text(0.05, 0.85,'Number of Units in RNN Cell: ' + str(n_cell), ha='left', va='center', color='blue', transform=ax.transAxes)
        text_acc = ax.text(0.65, 0.90,'Accuracy: ' + str(acc), ha='left', va='center', color='green', transform=ax.transAxes)
#        ax.text(0.65, 0.85, str(mnist.test.labels.shape[0])+ ' Samples', ha='left', va='center', color='green', transform=ax.transAxes)
        fig1.suptitle('Tensorflow RNN BasicLSTMCell - MNIST Digit Recognizer')
        plt.draw()
        plt.pause(0.3)
        
        try:
          
#          while step * batch_size < training_iters:

          submit_file = open(FLAGS.data_dir+submit_result_file_name, 'w+')
          submit_file.write("ImageId,Label\r\n")
          csv_w_submit = csv.writer(submit_file)
          submit_size = 0
 #         training_iters = 3000
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
                  print(submit_size)
                  show_mnist(feed_d[x],digits, "Outputs to submission file")
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
                  acc = np.mean(acc_l);
                           
                  show_mnist(s[1:],d[1:],"Incorect Classifications")
                  print(acc)
                  plt.figure(fig1.number)
                  plt.plot([step-test_at,step], [tmp, acc],'g')
                  text_acc.set_text('Accuracy: ' + str(acc))    
                  plt.draw()
                  plt.savefig(FLAGS.result_dir+'/mnist_rnn_LSTM'+'.png')
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

        result_str = str(round(int(acc*1000)))+'_step'+str(int(round(n_steps*1000)))+'_cell'+str(n_cell)+'_b'+str(step)
        plt.figure(fig1.number)
        plt.savefig(FLAGS.result_dir+'/mnist_rnn_LSTM'+result_str+'.png')
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
    
    train()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type=float, default=0.005,
                      help='Initial learning rate')
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
    
