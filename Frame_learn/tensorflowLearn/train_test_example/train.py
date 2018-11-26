from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import binascii
import time
 
#dataset
train_path='/home/audr/lhy/dataset/result_csv/bi_class_train_cnn_10percent.csv'
val_path='/home/audr/lhy/dataset/result_csv/bi_class_test_cnn_10percent.csv'
#saved model
model_path='save/model.ckpt'

#label_dict = {'IRC_attack':0,'Neris':1,'RBot':2,'Menti':3,'Sogou':4,'Murlo':5,'Virut':6,'Black_hole':7,'TBot':8,'Weasel':9,'Zeus':10,'Zero_access':11} 
label_dict = {'Normal':0,'Attack':1} 
#resize vector into 512*1
w=512
h=1
c=1


def getMatrixfrom_bin(hexst):
    f=[[]]
    fh = np.array([int(hexst[i:i+2],16) for i in range(0, len(hexst), 2)])
    f[0] = fh
    f = np.uint8(f)
    return f

def load_data(path):
    data=[]
    labels=[]
    f = open(path,'r')
    lines = f.readlines()        
    for line in lines: 
        try:
            item=getMatrixfrom_bin(line.strip().split(',')[0])
            item=transform.resize(item,(h,w,c))
            data.append(item)
            labels.append(label_dict[line.strip().split(',')[1]])
        except:
            print (line)

    return np.asarray(data,np.float32),np.asarray(labels,np.int32)
train_data,train_label=load_data(train_path)
val_data,val_label=load_data(val_path)
 
#shuffle pictures
num_example=train_data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
train_data=train_data[arr]
train_label=train_label[arr]
 
x_train=train_data
y_train=train_label
x_val=val_data
y_val=val_label

#----------------------------------------------network start
#
x=tf.placeholder(tf.float32,shape=[None,h,w,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')

def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight",[1,5,1,32],initializer=tf.truncated_normal_initializer(stddev=0.1))#kernel_h,kernel_w,in_channel,out_channel
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
 
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,1,2,1],strides=[1,1,2,1],padding="VALID")#[1,h,w,1][1,stride_h,stride_w,1]
 
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight",[1,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
 
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')
 
    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable("weight",[1,3,64,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
 
    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')
 
    with tf.variable_scope("layer7-conv4"):
        conv4_weights = tf.get_variable("weight",[1,3,128,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
 
    with tf.name_scope("layer8-pool4"):
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='VALID')
        print (pool4.shape)
        nodes = 32*1*128
        reshaped = tf.reshape(pool4,[-1,nodes])
 
    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))
 
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)
 
    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [1024, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))
 
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        print (fc2)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)
 
    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [512, 2],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [2], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases
    return logit
 
#----------------------------------------------network finish
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
logits = inference(x,False,regularizer)

#define logits_eval
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval') 

loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)    
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
 
#mini-batch
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]
 
 
#train and val, set n_epoch
 
n_epoch=10                                                                                                
batch_size=128
saver=tf.train.Saver()
sess=tf.Session()  
sess.run(tf.global_variables_initializer())
i=0
for epoch in range(n_epoch):
    start_time = time.time()
    i+=1
    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1
    print("   train loss: %f" % (np.sum(train_loss)/ n_batch))
    print("   train acc: %f" % (np.sum(train_acc)/ n_batch))
 
    #validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err; val_acc += ac; n_batch += 1
    print("   validation loss: %f" % (np.sum(val_loss)/ n_batch))
    print("   validation acc: %f" % (np.sum(val_acc)/ n_batch))
    saver.save(sess,model_path)
sess.close()
