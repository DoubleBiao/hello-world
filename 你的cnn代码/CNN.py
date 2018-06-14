# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:59:00 2018

@author: ZHANGSIJIA
"""

#加载必要的语言包
#必要的数据处理包Numpy，一般习惯缩写成np
import numpy as np 

#从Keras中加载必要的数据包
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

from slice_vector import slidingwindow_slice
#加载时间包
import time

#加载原始数据，分别为训练样本和测试数据。（根据自己的路径修改）

trian_x = np.load('train_x2.npy')    #train 写错啦！！！！ 
trian_y = np.load('train_y2.npy')
test_x = np.load('test_x2.npy')
test_y = np.load('test_y2.npy') 


#标准化，对数据原始训练和测试样本的进行标准化
trian_x = (trian_x-trian_x.min())/(trian_x.max()-trian_x.min())
test_x = (test_x-test_x.min())/(test_x.max()-test_x.min())

##########################改动在这儿######################
trian_x = slidingwindow_slice(trian_x,1,20,5000)
samplenum = len(trian_x)
trian_y = slidingwindow_slice(trian_y,1,1,samplenum)
trian_y = np.squeeze(trian_y)

test_x = slidingwindow_slice(test_x,1,20,1000)
samplenum = len(test_x)
test_y = slidingwindow_slice(test_y,1,1,samplenum)
test_y = np.squeeze(test_y)
##########################################################


#构建一个连续的CNN模型
model = Sequential()

#添加第一层卷积Convolution层
model.add(Convolution2D(256, 3, 3, border_mode='same', input_shape=(278, 20, 1)))
model.add(Dense(64, activation='tanh'))
#model.add(Activation(‘tanh‘))   #激活函数
model.add(MaxPooling2D(pool_size=(2, 2)))  #Pooling层

#添加第二层卷积Convolution层
model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(Dense(64, activation='tanh'))
#model.add(Activation(‘tanh’))  #激活函数
model.add(MaxPooling2D(pool_size=(2, 2))) #Pooling层

#添加一个展平层
model.add(Flatten())

#添加一个全连接层
model.add(Dense(278))
model.add(Activation("linear")) 

for layer in model.layers:
    print(layer.output_shape)
#添加编译器
model.compile(optimizer='rmsprop',loss='mse')

#设置时间函数
st = time.time()


    
#训练模型
model.fit(trian_x,trian_y,batch_size=20,epochs=20)#,validation_data=(test_x,test_y))
#训练模型评价
score = model.evaluate(trian_x, trian_y, verbose=1)
print('\n   Train',score)

#利用实验数据测试模型
score = model.evaluate(test_x, test_y, verbose=1)
print('\n   Test',score)
#输出运行时间
print('   Time', (time.time() - st) / 60.0)