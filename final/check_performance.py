import numpy
from skimage.measure import compare_ssim, compare_psnr
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

def PSNR_SSIM(y,x):
    (AMT,HT,WDTH) = y.shape
    PSNR_sum = 0
    SSIM_sum = 0
    (AMT, HT, WITH) = y.shape

    for i in range (AMT):
        y_tmp = y[i,:,:].reshape((HT,WDTH))
        x_tmp = x[i,:,:].reshape((HT,WITH))
        PSNR_sum += compare_psnr(x_tmp, y_tmp, data_range=1.0)
        SSIM_sum += compare_ssim(x_tmp, y_tmp, data_range=1.0)
    
    PSNR_sum = PSNR_sum/AMT
    SSIM_sum = SSIM_sum/AMT

    return (PSNR_sum, SSIM_sum)

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

def psnr_for2d(y, x):
    return tf.image.psnr(x, y, max_val=1.0)

def ssim_for2d(y, x):
    return tf.image.ssim(x, y, max_val=1.0)

def check_performance(x_set, y_set, HEIGHT, WIDTH, AMOUNT, without_padding):
    # only for x and y in the same shape
    if without_padding == True:
        x = numpy.reshape(x_set[:, 2:HEIGHT-2, 2:WIDTH-2, :],(AMOUNT, HEIGHT - 2*2, WIDTH- 2*2))
        y = numpy.reshape(y_set[:, 2:HEIGHT-2, 2:WIDTH-2, :],(AMOUNT, HEIGHT- 2*2, WIDTH- 2*2))
        print('x.shape:', x.shape, 'y.shape', y.shape)
    else:
        x = numpy.reshape(x_set[:, :, :, :],(AMOUNT, HEIGHT, WIDTH))
        y = numpy.reshape(y_set[:, :, :, :],(AMOUNT, HEIGHT, WIDTH))
        print('x.shape:', x.shape, 'y.shape', y.shape) 

    (psnr, ssim) = PSNR_SSIM(y,x)
    (mae, mse) = MAE_MSE(y,x)
    return (psnr, ssim, mae, mse)

def check_performance_3d(x_set, y_set, HEIGHT, WIDTH, AMOUNT):
    print('x_set.shape', x_set.shape)
    print('y_set.shape', y_set.shape)
    x = numpy.reshape(x_set[:, 2, 2:HEIGHT-2, 2:WIDTH-2, :],(AMOUNT - 6, HEIGHT - 2*2, WIDTH- 2*2))
    y = numpy.reshape(y_set[:, :, :, :],(AMOUNT - 6, HEIGHT- 2*2, WIDTH- 2*2))
    (psnr, ssim) = PSNR_SSIM(y,x)
    (mae, mse) = MAE_MSE(y,x)
    return (psnr, ssim, mae, mse)