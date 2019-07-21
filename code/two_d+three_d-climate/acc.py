from separator import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import tensorflow as tf
import keras.backend as K
import numpy
from PIL import Image
from skimage.measure import compare_ssim, compare_psnr

def renormalize(x):	#only used if dataset = climate
    norm = x / 20   #(-0.5, 0.5)
    norm = norm + 0.5
    return norm

#def renormalize(x):	#un-comment this function when dataset != climate
#    return x

def PSNR_SSIM(y,x):
    (AMT,HT,WDTH) = y.shape
    PSNR_sum = 0
    SSIM_sum = 0
    (AMT, HT, WITH) = y.shape

    for i in range (AMT):
        y_tmp = y[i,:,:].reshape((HT,WDTH))
        y_tmp = renormalize(y_tmp)
        x_tmp = x[i,:,:].reshape((HT,WITH))
        x_tmp = renormalize(x_tmp)
        PSNR_sum += compare_psnr(x_tmp, y_tmp, data_range=255)
        SSIM_sum += compare_ssim(x_tmp, y_tmp, data_range=255)
    
    PSNR_sum = PSNR_sum/AMT
    SSIM_sum = SSIM_sum/AMT

    return (PSNR_sum, SSIM_sum)

def psnr_for2d(y, x):
    #x = tf.dtypes.cast(x, tf.float32)
    #y = tf.dtypes.cast(y, tf.float32)
    (y, x) = (renormalize(y), renormalize(x))
    return tf.image.psnr(x, y, max_val=255)

def ssim_for2d(y, x):
    (y, x) = (renormalize(y), renormalize(x))
    return tf.image.ssim(x, y, max_val=255)

def MAE_MSE(y, x):
    (AMT, HT, WDTH) = y.shape
    mae = 0
    mse = 0
    for i in range(AMT):
        y_tmp = y[i, :, :].reshape((HT, WDTH))  
        y_tmp = renormalize(y_tmp)  
        x_tmp = x[i, :, :]. reshape((HT, WDTH))
        x_tmp = renormalize(x_tmp)
        
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
        y_tmp = renormalize(y_tmp)   
        x_tmp = x[i, :, :]. reshape((HT, WDTH))
        x_tmp = renormalize(x_tmp)
        
        acc_sum += accuracy_score(y_tmp, x_tmp)

    acc = acc_sum / AMT
    return acc

