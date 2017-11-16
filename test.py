# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#测试图片路径
image_path = './Test_image/'

#分类图片的具体类别名称
license_dict = {0: '0', 1: 'A', 2: '2', 3: 'B', 4: 'C', 5: '5', 6: '6', 7: 'D', 8: '8', 9: '鄂', 10: '京', 11: 'Q', 12: '渝'}

w = 47
h = 92
c = 1    #图片通道数

#函数，读取路径下面所有图片的名称，并保存在一个列表返回
def all_files(path):
    filename2 = os.walk(path)
    all_files = []
    for path, d, filename_list in filename2:
        for file_name in filename_list:
            one_file = os.path.join(path, file_name)
            all_files.append(one_file)
    return all_files

images = all_files(image_path)

#函数，读取一张图片，并向量化
def read_one_image(path):
    img = plt.imread(path)
    img = img.reshape([h,w,1])
    return np.asarray(img)

#对要测试的所有图片保存为一个矩阵
data = []
for i in range(len(images)):
    data.append(read_one_image(images[i]))
data = np.asarray(data)
#print(data.shape)

#函数， 依次显示所有测试图片
def image_show(imageset):
    for i in range(len(imageset)):
        image = imageset[i]
        #print(image.shape)
        plt.figure()
        string = 'this is NO.'+ str(i+1)+ ' test image'
        plt.title(string)
        plt.imshow(image[:,:,0], cmap='gray')
        plt.show()

image_show(data)

#函数，把经过模型识别输出的每一类的值转化为每一类的概率
def probability(data):
    aa = np.exp(data)
    bb = np.transpose(np.tile(np.sum(aa, axis=1), (data.shape[0], 1)))
    return aa/bb

#运行
with tf.Session() as sess:
	#找到我们训练好的模型
    saver = tf.train.import_meta_graph('model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x_tensor:0")		#找到模型的入口

    feed_dict = {x: data}			#输入需要测试的数据图片					

    logits = graph.get_tensor_by_name("logits_eval:0")		#模型的出口

    classification_result = sess.run(logits, feed_dict)

    # 打印出预测概率矩阵
    print('打印出预测概率矩阵')
    print(probability(classification_result))
    # 打印出预测矩阵每一行最大值的索引
    print('打印出预测矩阵每一行最大值的索引')
    print(tf.argmax(classification_result, 1).eval())
    # 根据索引通过字典对应花的分类
    print('根据索引通过字典对应花的分类')
    output = []
    output = tf.argmax(classification_result, 1).eval()
    for i in range(len(output)):
        print("第", i + 1, "图片预测结果是:" + license_dict[output[i]])