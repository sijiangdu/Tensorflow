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
# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
 
training_iters = 300000
batch_size = 100
sample_size = 28*28
n_cell= 400
n_steps = 14    # time steps
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
    outputs, final_state = tf.nn.dynamic_rnn(cell, X, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], W) + b     
    return results

# display image
show_num = 4
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
  
def train():
    
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
    
    with tf.Session() as sess:

        # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)
        step = 0
        acc = 0
        # plotting
        fig1, ax = plt.subplots(1,1)
        plt.ylim((-0.5, 1.5))
        plt.ylabel('Accuracy')
        plt.xlabel('Training Step ( bactch size:' + str(int(batch_size)) + ')')
        ax.text(0.05, 0.95,'Max Time -Steps in RNN cell: ' + str(n_steps), ha='left', va='center', color='blue', transform=ax.transAxes)
        ax.text(0.05, 0.90,'Number of Units in RNN Cell: ' + str(n_cell), ha='left', va='center', color='blue', transform=ax.transAxes)
        text_acc = ax.text(0.05, 0.85,'Accuracy: ' + str(acc), ha='left', va='center', color='green', transform=ax.transAxes)
        fig1.suptitle('Tensorflow RNN BasicLSTMCell - MNIST Digit Recognizer')
        acc_all = [0.0,0.0,0.0]
        plt.draw()
        plt.pause(0.3)
        
        while step * batch_size < training_iters:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, n_steps, -1])  
            sess.run([train_op], feed_dict={
                x: batch_xs,
                y: batch_ys,
            })
            step += 1
            
            #plotting
            test_at = 100
            if step % test_at == 0:
                tmp = acc
                #Init images those are incorrectly classified
                s = np.zeros((1,28*28))
                d = np.zeros(1)
                test_loop = 100
                acc_l = [0.0]*test_loop
                for i in range(test_loop):
                    batch_xs, batch_ys = mnist.test.next_batch(batch_size)
                    batch_xs = batch_xs.reshape([batch_size, n_steps, -1])  
                    acc_l[i],summary,digits = sess.run([accuracy,merged,classification], feed_dict={
                        x: batch_xs,
                        y: batch_ys,
                    })
                    train_writer.add_summary(summary, step)
                    
                    #store the images those are incorrectly classified
                    if len(s) > show_num*show_num: continue
                    correct = np.argmax(batch_ys,1)
                    for i in range(batch_size):
                      if correct[i] != digits[i]:
                        s = np.append(s,np.array([batch_xs[i].flatten()]),0)
                        d = np.append(d,np.array([digits[i]]),0)
                acc = np.mean(acc_l);
                #show the images those are incorrectly classified        
                show_mnist(s[1:],d[1:],"Incorect Classifications")                  
                print(acc)
                plt.figure(fig1.number)
                plt.plot([step-test_at,step], [tmp, acc],'g')
                acc_all.append(acc)
                acc_all.pop(0)
                text_acc.set_text('Accuracy:  ..., [%s], [%s], [%s]'%(acc_all[-3], acc_all[-2],acc_all[-1]) )    
                plt.draw()
                plt.savefig(FLAGS.result_dir+'/mnist_rnn_LSTM'+'.png')
                plt.pause(0.3)            
            
        result_str = str(round(int(acc*1000)))+'_step'+str(int(n_steps))+'_cell'+str(n_cell)+'_batch'+str(step)
        plt.figure(fig1.number)
        plt.savefig(FLAGS.result_dir+'/mnist_rnn_LSTM'+result_str+'.png')
        train_writer.close()


def main(_):
    
    if tf.gfile.Exists(FLAGS.log_dir):  tf.gfile.DeleteRecursively(FLAGS.log_dir)

    tf.gfile.MakeDirs(FLAGS.log_dir)
  
    if not tf.gfile.Exists(FLAGS.result_dir):   tf.gfile.MakeDirs(FLAGS.result_dir)
    
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
    
