from acc import *
from getpath import *
from load_data import *
from load_parameter import *
import keras
from keras.models import load_model
import os
'''
#------------when running cpu, block comment this session------------------------#
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#----------------------------------------------------------------------------------#
'''
#model_2d = keras.models.load_model('2d_model.h5', custom_objects={'ssim_for2d':ssim_for2d, 'psnr_for2d':psnr_for2d})
model_3d = keras.models.load_model('/home/user1/REUS/image-reconstruction/code/two_d+three_d/checkpoint.h5', custom_objects={'ssim_for2d':ssim_for2d, 'psnr_for2d':psnr_for2d})

(PATH, AMOUNT) = container()
AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, DEPTH = 300, 300, 176, 144, 352, 288, 5   #default
#(x_set, y_set, TARGET_HEIGHT, TARGET_WIDTH, HEIGHT, WIDTH) = container_2D(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH) #load for 2d
x_set = numpy.array([pickFrame('x', i, PATH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, False) for i in getIndex(AMOUNT, TOTAL_AMOUNT)])
x_set = numpy.reshape(x_set, (AMOUNT, TARGET_HEIGHT, TARGET_WIDTH, 1))
y_set = numpy.array([pickFrame('y', i, PATH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, False) for i in getIndex(AMOUNT, TOTAL_AMOUNT)])
y_set = numpy.reshape(y_set, (AMOUNT, TARGET_HEIGHT, TARGET_WIDTH, 1))
x = numpy.reshape(x_set[2:TOTAL_AMOUNT-2, 2:TARGET_HEIGHT - 2, 2:TARGET_WIDTH - 2, :], (TOTAL_AMOUNT - 4, TARGET_HEIGHT - 4, TARGET_WIDTH - 4))
y = numpy.reshape(y_set[2:TOTAL_AMOUNT-2, 2:TARGET_HEIGHT - 2, 2:TARGET_WIDTH - 2, :], (TOTAL_AMOUNT - 4, TARGET_HEIGHT - 4, TARGET_WIDTH - 4))
(psnr_raw, ssim_raw) = PSNR_SSIM(y, x)
(mae_raw, mse_raw) = MAE_MSE(y, x)
tmp_str = 'x.shape:'+ str(x.shape) + ' y.shape' + str(y.shape)
tmp_str_2 = 'psnr_raw, ssim_raw, mae_raw, mse_raw: '+ str(psnr_raw) + ', ' +str(ssim_raw) +', ' + str(mae_raw) +', ' + str(mse_raw) +'\n\n'


#mse_interim, ssim_interim, psnr_interim = model_2d.evaluate(x_set, y_set, verbose=2)
p_set = numpy.array([pickFrame('p', i, prediction_path(), HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, False) for i in getIndex(AMOUNT, TOTAL_AMOUNT)])
p_set = numpy.reshape(p_set, (AMOUNT, TARGET_HEIGHT, TARGET_WIDTH, 1))
p = numpy.reshape(p_set[2:TOTAL_AMOUNT-2, 2:TARGET_HEIGHT - 2, 2:TARGET_WIDTH - 2, :], (TOTAL_AMOUNT - 4, TARGET_HEIGHT - 4, TARGET_WIDTH - 4))
(psnr_interim, ssim_interim) = PSNR_SSIM(y, p)
(mae_interim, mse_interim) = MAE_MSE(y, p)
tmp_str_3 = 'p.shape: '+ str(p.shape) + ' y.shape: ' + str(y.shape)
tmp_str_4 = 'psnr_interim, ssim_interim, mae_interim, mse_interim: '+ str(psnr_interim) +', ' + str(ssim_interim) +', ' + str(mae_interim) +', ' + str(mse_interim) +'\n\n'
print(tmp_str)
print(tmp_str_2)
print(tmp_str_3)
print(tmp_str_4)
print('before processing package')

(AMOUNT, x_interim, HEIGHT, WIDTH) = prediction_package(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, DEPTH)    #load for 3d
x_final = model_3d.predict(x_interim)
#predict_y_set = y_set[2:TOTAL_AMOUNT-2, 2:TARGET_HEIGHT-2, 2:TARGET_WIDTH-2, :]
#y_set = predict_y_set
print('x_set.shape: ',x_set.shape, 'y_set.shape: ', y_set.shape, 'x_interim.shape: ', x_interim.shape)
f = x_final.reshape((AMOUNT, TARGET_HEIGHT - 4, TARGET_WIDTH - 4))
(psnr_final, ssim_final) = PSNR_SSIM(y, f)
(mae_final, mse_final) = MAE_MSE(y, f)
#mse_final, ssim_final, psnr_final = model_3d.evaluate(x_interim, y_set, verbose=2)

print(tmp_str)
print(tmp_str_2)
print(tmp_str_3)
print(tmp_str_4)
print('AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, DEPTH: ', AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, DEPTH)
print('psnr_final, ssim_final, mae_final, mse_final: ', psnr_final, ssim_final, mae_final, mse_final)
