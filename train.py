# -*- coding: utf-8 -*-

#导入程序运行需要的模块
import os							#读取文件
import numpy as np					#矩阵运算
import tensorflow as tf				#构建深度学习网络模块
import matplotlib.pyplot as plt		#显示、读取图片

image_width = 47					#图片宽度
image_height = 92					#图片高度
classes = 13						#要分类的类别数
index_length = image_height * image_width			#图片向量化后的长度
model_path = './model.ckpt'							#保存训练完成后模型的名字

# 第一次遍历图片目录是为了获取图片总数
input_count = 0
dir = './Mask_Image'
for root, dirs, files in os.walk(dir):
    for name in files:
        path = os.path.join(root, name)
        input_count += 1

# 定义对应维数和各维长度的数组
input_images = np.array([[0] * index_length for i in range(input_count)])		#建立保存图片数据集格式
input_labels = np.array([[0] * classes for i in range(input_count)])			#对应图片的标签

# 第二次遍历图片目录是为了生成图片数据和标签
index = 0
for i in range(classes):
    dir = dir = './Mask_Image/%s/' % i  # 这里可以改成你自己的图片目录，i为分类标签
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            filename = dir + filename
            img = plt.imread(filename)			#读取一张图片
            width = img.shape[1]
            height = img.shape[0]
            #print(width)
            #print(height)
            for h in range(height):
                for w in range(width):
                    input_images[index][w + h*width] = img[h][w]		#制作一张图片的数据集
            input_labels[index][i] = 1									#对应一张图片的标签
            index += 1


# 打乱数据集顺序，使训练不会发生震荡
num_example = input_images.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
input_images = input_images[arr]
input_labels = input_labels[arr]

#将所有数据分为训练集和验证集，本程序全部图片都用来训练，后面测试时用的单独的其他图片
ratio=1
s=np.int(num_example*ratio)
train_images = input_images[:s]
train_labels = input_labels[:s]
test_images = input_images[s:]
test_labels = input_labels[s:]

# 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
x = tf.placeholder(tf.float32, shape=[None, index_length],name='x')
y_ = tf.placeholder(tf.int32, shape=[None, classes],name='y_')

x_image = tf.reshape(x, [-1, image_height, image_width,  1],name='x_tensor')

# ---------------------------构建深度学习网络---------------------------

# 第1个卷积层(47->23, 92->46)
conv1 = tf.layers.conv2d(
    inputs=x_image,
    filters=32,					#卷积核的数量
    kernel_size=[5, 5],			#卷积核大小5x5
    padding="same",				#same表示卷积之后保持图片大小不变
    activation=tf.nn.relu,		#激活函数relu
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)) #初始化
#池化层，缩小图片尺寸
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# 第2个卷积层(23->11, 46->23)
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# 第3个卷积层(11->5, 23->11)
conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

# 第4个卷积层(5->2, 11->5)
conv4 = tf.layers.conv2d(
    inputs=pool3,
    filters=128,
    kernel_size=[3, 3],
    padding="same",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

re1 = tf.reshape(pool4, [-1, 2 * 5 * 128])

# dropout层，防止训练模型过拟合
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(re1, keep_prob=0.5)

# 全连接层
dense1 = tf.layers.dense(inputs=h_fc1_drop,
                         units=1024,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))   #正则化，防止模型过拟合
dense2 = tf.layers.dense(inputs=dense1,
                         units=512,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

logits = tf.layers.dense(inputs=dense2,units=13, name='logits')		#输出层，13个神经元，表示分13类

# ---------------------------网络结束---------------------------

b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval')

# 定义优化器和训练op
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_input_count = train_images.shape[0]
    print("一共读取了 %s 个输入图像， %s 个标签" % (train_input_count, train_input_count))

    # 设置每次训练op的输入个数和迭代次数，这里为了支持任意图片总数，定义了一个余数remainder，譬如，如果每次训练op的输入个数为60，图片总数为150张，则前面两次各输入60张，最后一次输入30张（余数30）
    batch_size = 30		#每次训练的batch数量30
    iterations = 50		#数据集迭代训练50次
    batches_count = int(train_input_count / batch_size)
    remainder = train_input_count % batch_size
    print("数据集分成 %s 批, 前面每批 %s 个数据，最后一批 %s 个数据" % (batches_count + 1, batch_size, remainder))

    # 执行训练迭代
    for it in range(iterations):
        # 这里的关键是要把输入数组转为np.array
        for n in range(batches_count):
			#把数据集分块进行训练，可以加快速度
            input_images_batch = train_images[n * batch_size:(n + 1) * batch_size]
            input_labels_batch = train_labels[n * batch_size:(n + 1) * batch_size]

            input_images_batch = np.asarray(input_images_batch)
            input_labels_batch = np.asarray(input_labels_batch)

            sess.run([train_step, cross_entropy, accuracy],
                     feed_dict={x: input_images_batch, y_: input_labels_batch, keep_prob: 0.5})

        if remainder > 0:
            start_index = batches_count * batch_size
            train_step.run(
                feed_dict={x: train_images[start_index:input_count - 1], y_: train_labels[start_index:input_count - 1], keep_prob: 0.5})

            # 每完成1次迭代，判断准确度是否已达到100%，达到则退出迭代循环
        iterate_accuracy = 0
        if it % 1 == 0:
            iterate_accuracy = accuracy.eval(feed_dict={x: train_images, y_: train_labels, keep_prob: 0.5})
            print('iteration---> %d:    accuracy---> %s' % (it, iterate_accuracy))
            if iterate_accuracy >= 1:
                break

#保存训练模型
    saver = tf.train.Saver()
    saver.save(sess, save_path=model_path)
    print('完成训练!')
    sess.close()