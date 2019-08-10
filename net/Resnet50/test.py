import tensorflow as tf
import pandas as pd
import numpy as np
import os
from skimage import io, transform
import glob
import time
import matplotlib.pyplot as plt
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
w=200
h=200
c=3

test_ = np.load("test_data.npz")
test_images = test_["images"]
test_visibility = test_["visibility"]
print(test_images.shape,test_visibility.shape)

x_test = test_images
y_test = test_visibility[:, np.newaxis]
print(x_test.shape,y_test.shape)


num_picture = 39
with tf.Session() as sess:
    #print(data.shape)
    #print(visibility.shape)
    saver = tf.train.import_meta_graph('./resnet_model/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./resnet_model/'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    print(y_)
    logits = graph.get_tensor_by_name("logits_eval:0")
    loss = tf.divide(tf.abs(y_ - logits),y_)
    test_loss, n_batch = 0, 0
    mape = []
    YY = []
    yy = []
    for i in range(num_picture):
        #y_train_a = y_train_a[:, np.newaxis]
        err,yuce, zhi = sess.run([loss, logits, y_], feed_dict={x: x_test[i:i+1], y_: y_test[i:i+1]})
        test_loss += err;
        mape.append(err[0,0])
        YY.append(zhi[0,0])
        yy.append(yuce[0,0])
        n_batch += 1
    testLoss = np.sum(test_loss) / n_batch
    print('__________________________')
    print("   test loss: %f" % (testLoss))
    print('真实值为：', YY)
    print('预测为:', yy)
    print(mape)
    #plt.plot(mape)
    #plt.title('resnet_average_mape:'+ str(testLoss))
    #plt.savefig('./resnet_mape.png')