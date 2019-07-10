import tensorflow as tf
import math

def to_ndarray(train, test):
    return (tf.Session().run(train), tf.Session().run(test))

def separator(mode, dataset, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT):
    
    if dataset.ndim == 4:
        #2d array
        return separator_2d(dataset, FRACTION, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)
    elif dataset.ndim == 5:
        if mode == 'y':
            TARGET_HEIGHT = TARGET_HEIGHT - 2* 2
            TARGET_WIDTH = TARGET_WIDTH - 2* 2
        #3d array
        return separator_3d(dataset, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)

def separator_2d(dataset, FRACTION, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT):
    
    TRAIN_AMOUNT = math.floor(FRACTION * AMOUNT)

    train = tf.slice(dataset,[0,0,0,0],[TRAIN_AMOUNT, TARGET_HEIGHT, TARGET_WIDTH,1])
    test = tf.slice(dataset,[TRAIN_AMOUNT,0,0,0],[AMOUNT - TRAIN_AMOUNT, TARGET_HEIGHT, TARGET_WIDTH,1])

    return to_ndarray(train, test)

def separator_3d(dataset, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT):
    
    TRAIN_AMOUNT = math.floor(FRACTION * AMOUNT)

    train = tf.slice(dataset,[0,0,0,0,0],[TRAIN_AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH,1])
    test = tf.slice(dataset,[TRAIN_AMOUNT,0,0,0,0],[AMOUNT - TRAIN_AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH,1])

    return to_ndarray(train, test)

