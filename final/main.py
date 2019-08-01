# -*- coding: utf-8 -*-
from load_data import *
from check_performance import *
from code_3d import three_d
from code_3d_processed import three_d_processed
from two_d import two_d
from two_d_three_d import two_d_three_d
from two_d_processed_three_d import two_d_processed_three_d
from fsrcnn import fsrcnn
import numpy
import os

'''
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))
'''

#------------when running cpu, block comment this session------------------------#
#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#---------------------------------------------------------------------------------#


DATA_FILE = '/home/user1/REUS/data_backup/data/climate_data.h5' 
START_INDEX, END_INDEX = 650, 700
AMOUNT = 120    #load the latest 300 frames from the 1000 frames

DEPTH = 5

(train_x, AMOUNT, HEIGHT, WIDTH) = load_h5_data(DATA_FILE, 'train', AMOUNT)
(train_y, AMOUNT, TARGET_HEIGHT, TARGET_WIDTH) = load_h5_data(DATA_FILE, 'label', AMOUNT)

(test_x, AMOUNT_TEST, HEIGHT, WIDTH) = load_h5_data_testset(DATA_FILE, 'train', START_INDEX, END_INDEX)
(test_y, AMOUNT_TEST, TARGET_HEIGHT, TARGET_WIDTH) = load_h5_data_testset(DATA_FILE, 'label', START_INDEX, END_INDEX)

#'HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH:', 159 320 318 640

#2d model with transfer learning
#model_2d = two_d(train_x, train_y, test_x, test_y, AMOUNT, AMOUNT_TEST, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)

#fsrcnn model
model_fsrcnn = fsrcnn(train_x, train_y, test_x, test_y, AMOUNT, AMOUNT_TEST, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
print('finish fsrcnn model training')

#3d model
model_3d = three_d(train_x, train_y, test_x, test_y, AMOUNT, AMOUNT_TEST, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
print('finish 3d model training')

#fsrcnn + 3d model
model_2d_3d = two_d_three_d(train_x, train_y, test_x, test_y, AMOUNT, AMOUNT_TEST, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
print('finish 2d_3d model training')

#3d model trained with input processed
model_3d_processed = three_d_processed(train_y, test_y, AMOUNT, AMOUNT_TEST, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
print('finish 3d_processed model training')

#fsrcnn + 3d model trained with input processed
model_2d_processed_3d = two_d_processed_three_d(train_x, train_y, test_x, test_y, AMOUNT, AMOUNT_TEST, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
