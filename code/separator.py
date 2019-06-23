import tensorflow as tf
import math

def separator(dataset, FRACTION, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT):
    
    TRAIN_AMOUNT = math.floor(FRACTION * AMOUNT)

    train = tf.slice(dataset,[0,0,0,0],[TRAIN_AMOUNT, TARGET_HEIGHT, TARGET_WIDTH,1])
    test = tf.slice(dataset,[TRAIN_AMOUNT,0,0,0],[AMOUNT - TRAIN_AMOUNT, TARGET_HEIGHT, TARGET_WIDTH,1])

    return to_ndarray(train, test)
'''
def norm(x, train_dataset):   
    mean_stat = tf.math.reduce_mean(tf.cast(train_dataset, tf.float32), 0)
    std_stat = tf.math.reduce_std(tf.cast(train_dataset, tf.float32), 0)

    return (x - mean_stat) / std_stat

def normalize(train, test):
    
    (train, test) = (norm(tf.cast(train, tf.float32), train), norm(tf.cast(test, tf.float32), train))
    
    return to_ndarray(train, test)
'''
def to_ndarray(train, test):
    return (tf.Session().run(train), tf.Session().run(test))
