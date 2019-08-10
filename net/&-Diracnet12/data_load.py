import pandas as pd
import numpy as np
import os
from skimage import io, transform
def zc_read_csv():
    zc_dataframe = pd.read_csv("./rjc_1min.csv", sep=",")
    x = []
    y = []
    #print(zc_dataframe)
    for zc_index in zc_dataframe.index:
        zc_line = zc_dataframe.loc[zc_index]
        x.append(zc_line["address"])
        y.append(zc_line["visibility"])
    return  np.asarray(x,np.str), np.asarray(y, np.float32)
address,kanjian= zc_read_csv()
print(address[0])
# 读取图片
w=200
h=200
c=3
train_path = './fog_1508_data/train/'
val_path = './fog_1508_data/val/'
test_path = './fog_1508_data/test/'
# 读取图片
def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path)]  # 列表生成式
    cate = list(sorted(cate, key=lambda x: int(x.split("_")[-1].split(".")[0])))
    imgs = []
    labels = []
    visibilitys = []
    for idx, folder in enumerate(cate):  # 标签排序
        cate = [folder+"/"+x for x in os.listdir(folder)]
        cate = list(sorted(cate, key=lambda x: int(x.split("_")[-1].split(".")[0])))
        for im in cate:  # glob返回文件名只包括当前目录里的文件名，不包括子文件夹里的文件。
            print('reading the images:%s' % (im))
            i = address.tolist().index(im)
            shuzi = kanjian[i]
            img = io.imread(im)
            img = transform.resize(img, (w, h))
            imgs.append(img)  # append往空列表尾部插入元素
            labels.append(idx)
            visibilitys.append(shuzi)
    return np.asarray(imgs, np.float32),np.asarray(visibilitys, np.float32)  # 将列表转换成数组

train_data, train_visibility = read_img(train_path)
val_data, val_visibility = read_img(val_path)
test_data, test_visibility = read_img(test_path)
np.savez("train_data.npz", images=train_data, visibility=train_visibility)
np.savez("val_data.npz", images=val_data, visibility=val_visibility)
np.savez("test_data.npz", images=test_data, visibility=test_visibility)