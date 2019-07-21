import tensorflow as tf
import math
import numpy as np

def to_ndarray(train, test):
    return (tf.Session().run(train), tf.Session().run(test))

def separator(dataset, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT):
    
    if dataset.ndim == 4:
        #2d array
        (tmp1, TARGET_HEIGHT, TARGET_WIDTH, tmp2) = dataset.shape
        return separator_2d(dataset, FRACTION, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)
    elif dataset.ndim == 5:
        (tmp1, tmp3, TARGET_HEIGHT, TARGET_WIDTH, tmp2) = dataset.shape
        #3d array
        return separator_3d(dataset, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)
'''
def separator_2d(dataset, FRACTION, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT):
    
    TRAIN_AMOUNT = math.floor(FRACTION * AMOUNT)

    train = tf.slice(dataset,[0,0,0,0],[TRAIN_AMOUNT, TARGET_HEIGHT, TARGET_WIDTH,1])
    test = tf.slice(dataset,[TRAIN_AMOUNT,0,0,0],[AMOUNT - TRAIN_AMOUNT, TARGET_HEIGHT, TARGET_WIDTH,1])

    return to_ndarray(train, test)
'''
def separator_2d(dataset, FRACTION, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT):
    
    TRAIN_AMOUNT = math.floor(FRACTION * AMOUNT)

    train = dataset[0:TRAIN_AMOUNT, :, :, :]
    test = dataset[TRAIN_AMOUNT:AMOUNT, :, :, :]
    return (train, test)

def separator_3d(dataset, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT):
    
    TRAIN_AMOUNT = math.floor(FRACTION * AMOUNT)

    #train = tf.slice(dataset,[0,0,0,0,0],[TRAIN_AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH,1])
    #test = tf.slice(dataset,[TRAIN_AMOUNT,0,0,0,0],[AMOUNT - TRAIN_AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH,1])

    train = dataset[0:TRAIN_AMOUNT, :, :, :, :]
    test = dataset[TRAIN_AMOUNT:AMOUNT, :, :, :, :]

    return (train, test)

