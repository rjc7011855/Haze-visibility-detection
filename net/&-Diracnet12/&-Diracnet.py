import numpy as np
import tensorflow as tf
from dirac_layers import *
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
w=200
h=200
c=3
start = time.time()
print('_________')
print('start:',start)
train_ = np.load("train_data.npz")
val_ = np.load("val_data.npz")
train_images = train_["images"]
train_visibility = train_["visibility"]
val_images = val_["images"]
val_visibility = val_["visibility"]
print(train_images.shape,train_visibility.shape)

# 打乱顺序
num_example = train_images.shape[0]     #shape 矩阵维度
arr = np.arange(num_example)
np.random.shuffle(arr)
data = train_images[arr]
visibility = train_visibility[arr]



x_train = train_images
y_train = train_visibility[:, np.newaxis]
print(x_train.shape,y_train.shape)
x_val = val_images
y_val = val_visibility[:, np.newaxis]

x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.float32,shape=[None,1],name='y_')

def inference (input_tensor):
    with tf.variable_scope('conv1'):
        conv1_weights = tf.get_variable("weight", [7, 7, 3, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.0001))
        conv1_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 2, 2, 1], padding='VALID')
        conv1 = tf.layers.batch_normalization(conv1, axis=3)
        x = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    outdim = 16
    for group in range(0, 3):
        for block in range(0, 2):
            #x = ncrelu(x, name="crelu_" + str(group) + "_" + str(block))
            x = dirac_conv2d(x, outdim, 3, 3, 1, 1, name="conv_" + str(group) + "_" + str(block))
            x = dirac1_conv2d(x, outdim, 3, 3, 1, 1, name="conv1_" + str(group) + "_" + str(block))
            x = ncrelu(x, name="crelu_" + str(group) + "_" + str(block))
        if (group != 3 - 1):
            x = tf.nn.pool(x, [2, 2], "MAX", "VALID", None, [2, 2], name="maxpool_" + str(group))

        outdim = outdim * 2

    temp_shape = x.get_shape().as_list()
    x = tf.nn.avg_pool(x, [1, temp_shape[1], temp_shape[2], 1], [1, temp_shape[1], temp_shape[2], 1],
                               "VALID", name="avgpool")
    flatten = tf.contrib.layers.flatten(x)
    dense1 = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
    logits = tf.layers.dense(dense1, units=1, activation=None)
    return dense1,logits

dense1,logits = inference(x)
b = tf.constant(value=1,dtype=tf.float32)
dense1_eval = tf.multiply(dense1,b,name='dense1_eval')
logits_eval = tf.multiply(logits,b,name='logits_eval')
loss = tf.reduce_mean(tf.square(y_ - logits))

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

a = tf.abs(tf.subtract(logits,y_))
d = y_*0.1
correct_prediction=tf.less_equal(a,d)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
mape = tf.reduce_mean(tf.divide(tf.abs(y_ - logits),y_))

def shuffle_data(inputs=None, targets=None, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    return inputs[indices], targets[indices]

model_path='./1.24&-dirac_model/1.24model.ckpt'
# 训练和测试数据，可将n_epoch设置更大一些
n_epoch = 100
batch_size = 64
loss_basic = 100000
num = 0
loss_end = 0
mape_end = 0
acc_end = 0
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(n_epoch):
    x_train_all, y_train_all = shuffle_data(x_train, y_train)
    print(x_train_all.shape, y_train_all.shape)

    train_loss, train_mape, train_acc, n_batch = 0, 0, 0, 0

    for i in range(0, len(x_train_all) + 1, batch_size):
        start = i
        end = min(i+batch_size,len(x_train))
        x_train_a, y_train_a = x_train_all[start:end], y_train_all[start:end]
        #print(x_train_a.shape, y_train_a.shape)
        _, err, ac, mape_loss, yuce, zhi = sess.run([train_op, loss, acc, mape, logits, y_], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err;
        train_acc += ac;
        train_mape += mape_loss
        n_batch += 1
        print('epoch ' + str(epoch) + ', batch ' + str(n_batch) )
        print("   this batch train loss: %f" % (err))
        print("   this batch train mape: %f" % (mape_loss))
        print("   this batch train acc: %f" % (ac))
        print('真实值为:', zhi)
        print('预测值为：',yuce)
    trainLoss = train_loss / n_batch
    trainMape = train_mape / n_batch
    trainAcc = train_acc / n_batch
    print("epoch " + str(epoch) + " train loss: %f" % (trainLoss))
    print("epoch " + str(epoch) + " train mape: %f" % (trainMape))
    print("epoch " + str(epoch) + " train acc: %f" % (trainAcc))

    # validation
    start2 = epoch * batch_size % len(x_val)
    end2 = min(start2 + batch_size, len(x_val))
    #val_loss, val_acc, n_batch = 0, 0, 0
    #for batches in range(end2-start2):
    err, ac, mape_loss = sess.run([loss, acc, mape], feed_dict={x: x_val[start2:end2], y_: y_val[start2:end2]})
    print("   validation loss: %f" % (err))
    print("   validation mape: %f" % (mape_loss))
    print("   validation acc: %f" % (ac))
    print('epoches:', epoch)
    if err < loss_basic:
        loss_basic = err
        saver.save(sess, model_path)
        num = epoch
        loss_end = err
        mape_end = mape_loss
        acc_end = ac
print("epoches_end=", num)
print("loss_end=", loss_end)
print("mape_end=",mape_end)
print("acc_end=", acc_end)
sess.close()
finish = time.time()
print('finish:',finish)
TIME = finish-start
print('time:',TIME)