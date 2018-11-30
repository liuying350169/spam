# -*- coding: utf-8 -*-
"""
Created on March 8 , 2018

author: liuying
"""
import random
import time
import numpy as np
import tensorflow as tf
import datetime
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Lambda, Activation, Reshape, RepeatVector, Permute
from keras import optimizers
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras import backend as K
from keras.layers import *
from keras import regularizers
from keras.layers.merge import *
from keras.models import *
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import Bidirectional
from gensim.models import *

# load data
dataset_path = 'D:\pytorch\model\dataset/'
junk_list = []  # 垃圾词列表
weibo = []  # 微博语料
test = []  # 测试集
embed = Word2Vec.load(dataset_path + 'word2vec_wx')  # embedding 一个词对应后面256维的向量
with open(dataset_path + 'level_0.4_out_empty.txt', encoding='utf-8') as f:
    for line in f:
        junk_list.append(line.split('\n')[0])  # 添加

with open(dataset_path + 'level_0.5_out_empty.txt', encoding='utf-8') as f:
    for line in f:
        junk_list.append(line.split('\n')[0])  # 添加
#交叉验证
with open(dataset_path + 'part0.txt', encoding='utf-8') as f:  # 语料
    for line in f:
            test.append(line)
with open(dataset_path + 'part1.txt', encoding='utf-8') as f:  # 语料
    for line in f:
            weibo.append(line)
with open(dataset_path + 'part2.txt', encoding='utf-8') as f:  # 语料
    for line in f:
            weibo.append(line)
with open(dataset_path + 'part3.txt', encoding='utf-8') as f:  # 语料
    for line in f:
            weibo.append(line)
with open(dataset_path + 'part4.txt', encoding='utf-8') as f:  # 语料
    for line in f:
            weibo.append(line)
with open(dataset_path + 'part5.txt', encoding='utf-8') as f:  # 语料
    for line in f:
            weibo.append(line)
with open(dataset_path + 'part6.txt', encoding='utf-8') as f:  # 语料
    for line in f:
            weibo.append(line)
with open(dataset_path + 'part7.txt', encoding='utf-8') as f:  # 语料
    for line in f:
            weibo.append(line)
with open(dataset_path + 'part8.txt', encoding='utf-8') as f:  # 语料
    for line in f:
            weibo.append(line)
with open(dataset_path + 'part9.txt', encoding='utf-8') as f:  # 语料
    for line in f:
            weibo.append(line)



# 乱序
random.shuffle(weibo)
random.shuffle(test)

# length是测试集语料长度的
length = len(weibo)  # 语料库长度
# np.zeros 表示用0填充
x_train = np.zeros((length, 87, 256))  # 87是最长的一句的词数
# np.zeros((2, 1)) 表示一个三维数组 全0
# array([[ 0.],
#       [ 0.]])
# print(x_train)
y_train = np.zeros((length, 1))
# print(y_train)
x_junk = np.zeros((length, 16, 256))  # 32是同一条微博下带的最多的垃圾词有32条
# 三个三维数组
counttt = 0
# 训练集的处理
for i in range(length):
    words = weibo[i].split('\n')[0].split(' ')  # 按照\n 是下一条  空格是下一个词
    # print(weibo[i].split('\n')[0])把一句话取出来 本来是一维容器里的一句
    # print(weibo[i].split('\n'))带有容器的
    # print(words)一个个的词
    L = len(words)  # 一句有多少词
    tmpJunk = 0
    for j in range(L - 1):
        try:
            x_train[i][j] = embed[words[j]]  # i是第几条，j是第几个词，将embedding的值嵌入矩阵
            # print(embed[words[j]])#一个256维的向量
        except KeyError:
            pass

        if junk_list.count(words[j]) != 0:  # 垃圾词表 中 word[j]出现的次数 只要出现一次即可 word[j] 是第i条的第j个词
            try:
                # print('i',i)
                # print('j',j)
                # print('tmpJunk',tmpJunk)
                # if tmpJunk>15:
                #     tmpJunk = 0
                #     counttt=counttt+1
                #     print('counttt',counttt)
                # 目前仍在考虑第i条的第j个词  进入到这的原因是 第j个词在垃圾词表中，所以存入x_junk中，跟在第i条后面 tmpJunk表示第i条后面一共有多少个这样的垃圾词
                x_junk[i][tmpJunk] = embed[words[j]]  # 将词嵌入对应的垃圾词表  表示将第i条 后面的第tmpJunk个词嵌入256维的向量
                tmpJunk = tmpJunk + 1  # 记录用了多少个junk词
            except KeyError:
                pass
    y_train[i] = int(words[L - 1])  # 最后一个词---就是标签0或1  0 正常1负样  y_train的第i行记录 这条样本是正样本还是负样本

# 测试集的处理
# 测试集的长度有问题啊
test_length = len(test)
x_test = np.zeros((test_length, 87, 256))
y_test = np.zeros((test_length, 1))
test_junk = np.zeros((test_length, 16, 256))

for i in range(test_length):
    test_words = test[i].split('\n')[0].split(' ')
    test_L = len(test_words)
    tmpJunk = 0
    for j in range(test_L - 1):
        try:
            x_test[i][j] = embed[test_words[j]]
        except KeyError:
            pass

        if junk_list.count(test_words[j]) != 0:
            try:
                test_junk[i][tmpJunk] = embed[test_words[j]]
                tmpJunk = tmpJunk + 1
            except KeyError:
                pass
    y_test[i] = int(test_words[test_L - 1])

input_size = 256  # 词嵌入256维
step = 87  #
unit = 16  #


def foo(x):
    return K.sum(x, axis=1)


# 第1轴
def bar(x):
    return K.sum(x, axis=1) / 16


#
# 开始码神经网络
context = Input(shape=(step, input_size), name='context_input')  # 跟embedding相关的 shape(步长，inputsize)
junk = Input(shape=(16, input_size), name='junk_input')  # 垃圾词的shape 模型的另一端
context_mask = Masking(mask_value=0)(context)  # mask层处理变长 补0补齐之后 去掉 不影响向量计算
junk_mask = Masking(mask_value=0)(junk)  # 首先将序列转换为定长序列，如，选取一个序列最大长度，不足这个长度的序列补-1。然后在Masking层中mask_value中指定过滤字符。如上代码所示，序列中补的-1全部被过滤掉。
lstm_context = LSTM(units=unit, input_shape=(step, input_size), return_sequences=True, name='lstm_context')(context_mask)  # mask处理后的第一层LSTM
lstm_context = Dropout(0.5)(lstm_context)  # dropout我们在这里使用了dropout方法。所谓dropout,就是指网络中每个单元在每次有数据流入时以一定的概率(keep prob)正常工作，否则输出0值。这是是一种有效的正则化方法，可以有效防止过拟合。
lstm_junk = LSTM(units=unit, return_sequences=True, name='lstm_junk')(junk_mask)  # mask处理后的第一层LSTM
lstm_junk = Dropout(0.5)(lstm_junk)  # 我们在这里使用了dropout方法。所谓dropout,就是指网络中每个单元在每次有数据流入时以一定的概率(keep prob)正常工作，否则输出0值。这是是一种有效的正则化方法，可以有效防止过拟合。
lstm_junk = Lambda(bar)(lstm_junk)
lstm_junk = RepeatVector(step)(lstm_junk)  # RepeatVector层将输入重复n次

# 将context和junk拼接
lstm_merge = concatenate([lstm_context, lstm_junk])  # 该层接收一个列表的同shape张量，并返回它们的按照给定轴相接构成的向量。

lstm_attention = TimeDistributed(Dense(unit, activation='tanh', use_bias=False))(lstm_merge)  # 输入至少为3D张量，下标为1的维度将被认为是时间维；例如，考虑一个含有32个样本的batch，每个样本都是10个向量组成的序列，每个向量长为16，则其输入维度为(32,10,16)，其不包含batch大小的input_shape为(10,16)；我们可以使用包装器TimeDistributed包装Dense，以产生针对各个时间步信号的独立全连接：
lstm_attention = Dropout(0.5)(lstm_attention)  # 我们在这里使用了dropout方法。所谓dropout,就是指网络中每个单元在每次有数据流入时以一定的概率(keep prob)正常工作，否则输出0值。这是是一种有效的正则化方法，可以有效防止过拟合。
lstm_attention = TimeDistributed(Dense(unit, use_bias=False))(lstm_attention)
lstm_attention = Dropout(0.5)(lstm_attention)  # 我们在这里使用了dropout方法。所谓dropout,就是指网络中每个单元在每次有数据流入时以一定的概率(keep prob)正常工作，否则输出0值。这是是一种有效的正则化方法，可以有效防止过拟合。
lstm_attention = Activation('softmax')(lstm_attention)  # 需要确定LSTM模块的激活函数（activation fucntion）（keras中默认的是tanh）； #softmax也是一个函数
lstm_final_representation = Multiply()([lstm_context, lstm_attention])  # Multiply矩阵相乘
lstm_output = Lambda(foo)(lstm_final_representation)
lstm_output = Dense(1,activation='sigmoid')(lstm_output)  # 选择sigmoid作为激活函数    Dense就是常用的全连接层，所实现的运算是output = activation(dot(input, kernel)+bias)。其中activation是逐元素计算的激活函数，kernel是本层的权值矩阵，bias为偏置向量，只有当use_bias=True才会添加。

model = Model(inputs=[context, junk], outputs=[lstm_output])  # 构建模型 定义最初input 和 output

# 在训练模型之前，我们需要通过compile来对学习过程进行配置。compile接收三个参数：
# 优化器optimizer：该参数可指定为已预定义的优化器名，如rmsprop、adagrad，或一个Optimizer类的对象，详情见optimizers
# 损失函数loss：该参数为模型试图最小化的目标函数，它可为预定义的损失函数名，如categorical_crossentropy、mse，也可以为一个损失函数。详情见losses
# 指标列表metrics：对分类问题，我们一般将该列表设置为metrics=['accuracy']。指标可以是一个预定义指标的名字,也可以是一个用户定制的函数.指标函数应该返回单个张量,或一个完成metric_name - > metric_value映射的字典.请参考性能评估
model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())  # 打印模型的层结构

print('training---------------------------')
nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
print('\n',nowTime)
model.fit([x_train, x_junk], y_train, epochs=14, batch_size=1, validation_split=0.1)  # fit开始训练 #batch 几个一批
nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
print('\n',nowTime)
print('testing-----------------------------')
score = model.evaluate([x_test, test_junk], y_test)  # 开始评估
print(score)
nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
print('\n',nowTime)
#print('改回原版 6轮')
