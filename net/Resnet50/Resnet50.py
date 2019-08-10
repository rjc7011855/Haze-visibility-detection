import numpy as np
import tensorflow as tf
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

TRAINING = tf.Variable(initial_value=True, dtype=tf.bool, trainable=False)

def identity_block(X_input, kernel_size, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    with tf.name_scope("id_block_stage"+str(stage)):
        filter1, filter2, filter3 = filters
        X_shortcut = X_input
        x = tf.layers.conv2d(X_input, filter1,kernel_size=(1, 1), strides=(1, 1),name=conv_name_base+'2a')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base+'2a', training=TRAINING)
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, filter2, (kernel_size, kernel_size),padding='same', name=conv_name_base+'2b')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base+'2b', training=TRAINING)
        x = tf.nn.relu(x)

        # Third component of main path
        x = tf.layers.conv2d(x, filter3, kernel_size=(1, 1),name=conv_name_base+'2c')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2c', training=TRAINING)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X_add_shortcut = tf.add(x, X_shortcut)
        add_result = tf.nn.relu(X_add_shortcut)

    return add_result


def convolutional_block(X_input, kernel_size, filters, stage, block, stride = 2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    with tf.name_scope("conv_block_stage" + str(stage)):
        filter1, filter2, filter3 = filters

        X_shortcut = X_input

        x = tf.layers.conv2d(X_input, filter1,kernel_size=(1, 1),strides=(stride, stride),name=conv_name_base+'2a')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base+'2a', training=TRAINING)
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, filter2, (kernel_size, kernel_size), name=conv_name_base + '2b',padding='same')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2b', training=TRAINING)
        x = tf.nn.relu(x)

        x = tf.layers.conv2d(x, filter3, (1, 1), name=conv_name_base + '2c')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2c', training=TRAINING)

        X_shortcut = tf.layers.conv2d(X_shortcut, filter3, (1,1),
                                      strides=(stride, stride), name=conv_name_base + '1')
        X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=3, name=bn_name_base + '1', training=TRAINING)

        X_add_shortcut = tf.add(X_shortcut, x)
        add_result = tf.nn.relu(X_add_shortcut)
    return add_result

def inference(input_tensor):
    with tf.variable_scope('conv1'):
        conv1_weights = tf.get_variable("weight", [7,7, 3, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.0001))
        conv1_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 2, 2, 1], padding='VALID')
        conv1 = tf.layers.batch_normalization(conv1, axis=3)
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    with tf.name_scope("pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

    # stage 2
    x = convolutional_block(pool1, kernel_size=3, filters=[64, 64, 256], stage=2, block='a', stride=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # stage 3
    x = convolutional_block(x, kernel_size=3, filters=[128,128,512],stage=3, block='a', stride=2)
    x = identity_block(x, 3, [128,128,512], stage=3, block='b')
    x = identity_block(x, 3, [128,128,512], stage=3, block='c')
    x = identity_block(x, 3, [128,128,512], stage=3, block='d')

    # stage 4
    x = convolutional_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='a', stride=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # stage 5
    x = convolutional_block(x,kernel_size=3,filters=[512, 512, 2048], stage=5, block='a', stride=2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = tf.layers.average_pooling2d(x, pool_size=(2, 2), strides=(1,1))

    flatten = tf.contrib.layers.flatten(x)
    dense1 = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
    logits = tf.layers.dense(dense1, units=1, activation=None)
    return logits

logits = inference(x)
#(y_-logits)/y_
loss = tf.reduce_mean(tf.square(y_ - logits))
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval')
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

model_path='./1.24resnet_model/1.24model.ckpt'
# 训练和测试数据，可将n_epoch设置更大一些
n_epoch = 100
batch_size = 64
loss_basic = 100000
num = 0
loss_end = 0
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