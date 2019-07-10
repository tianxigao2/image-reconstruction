from separator import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import tensorflow as tf
import keras.backend as K
import numpy
from PIL import Image

def PSNR(y, x): #y_true, y_pred for 3d input
    (AMT, HT, WDTH) = y.shape
    PSNR_sum = 0
    for i in range(AMT):
        y_tmp = y[i, :, :].reshape((HT, WDTH)) 
        y_tmp = tf.dtypes.cast(y_tmp,tf.float32)
        
        x_tmp = x[i, :, :].reshape((HT, WDTH))
        x_tmp = tf.dtypes.cast(x_tmp,tf.float32) 
        
        (x_tmp, y_tmp) = to_ndarray(x_tmp, y_tmp)
        y_tmp = y_tmp.reshape((HT, WDTH, 1))
        x_tmp = x_tmp.reshape((HT, WDTH, 1))
        y_tmp = tf.convert_to_tensor(y_tmp, dtype = tf.float32)
        x_tmp = tf.convert_to_tensor(x_tmp, dtype = tf.float32)
        
        PSNR_sum += tf.Session().run(tf.image.psnr(x_tmp, y_tmp, max_val=1))
        print(PSNR_sum.shape)

    psnr = PSNR_sum/AMT    
    return psnr 

def psnr_for2d(y, x):
    #x = tf.dtypes.cast(x, tf.float32)
    #y = tf.dtypes.cast(y, tf.float32)
    return tf.image.psnr(x, y, max_val=1)

def SSIM(y, x):
    (AMT, HT, WDTH) = y.shape
    ssim_sum = 0
    for i in range(AMT):
        y_tmp = y[i,:,:].reshape((HT, WDTH))
        y_tmp = tf.dtypes.cast(y_tmp,tf.float32)

        x_tmp = x[i, :, :].reshape((HT, WDTH))
        x_tmp = tf.dtypes.cast(x_tmp,tf.float32)

        (x_tmp, y_tmp) = to_ndarray(x_tmp, y_tmp)
        y_tmp = y_tmp.reshape((HT, WDTH, 1))
        x_tmp = x_tmp.reshape((HT, WDTH, 1))
        y_tmp = tf.convert_to_tensor(y_tmp, dtype = tf.float32)
        x_tmp = tf.convert_to_tensor(x_tmp, dtype = tf.float32)

        ssim_sum += tf.Session().run(tf.image.ssim(x_tmp, y_tmp, max_val=1))    #Image.fromarray(x_tmp), Image.fromarray(y_tmp)
    
    ssim = ssim_sum/AMT

    return ssim

def ssim_for2d(y, x):
    return tf.image.ssim(x, y, max_val=1)

def MAE_MSE(y, x):
    (AMT, HT, WDTH) = y.shape
    mae = 0
    mse = 0
    for i in range(AMT):
        y_tmp = y[i, :, :].reshape((HT, WDTH))    
        x_tmp = x[i, :, :]. reshape((HT, WDTH))
        
        mae += mean_absolute_error(y_tmp, x_tmp)
        mse += mean_squared_error(y_tmp, x_tmp)

    mae = mae / AMT
    mse = mse / AMT

    return (mae, mse)

def acc(y, x):
    (AMT, HT, WDTH) = y.shape
    acc_sum = 0
    for i in range(AMT):
        y_tmp = y[i, :, :].reshape((HT, WDTH))    
        x_tmp = x[i, :, :]. reshape((HT, WDTH))
        
        acc_sum += accuracy_score(y_tmp, x_tmp)

    acc = acc_sum / AMT
    return acc
