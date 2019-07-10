from load_data import *
from separator import *
from model_build_up import *
from train import *
from getpath import *
from acc import *
import numpy
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.callbacks import EarlyStopping


'''
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))
'''

#------------when running cpu, block comment this session------------------------#
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#---------------------------------------------------------------------------------#

#=================================================================================
'''
# use shepp-logan-phantom for testing---------------------------------------------
AMOUNT = 2000    #might be changed after picking frames
TOTAL_AMOUNT = 2000

# change max_val in acc.py to 1
'''
# use any other set for testing -------------------------------------------------------
AMOUNT = 300
TOTAL_AMOUNT = 300

# max_val in acc.py is 255
#==================================================================================
#image size for data x
HEIGHT = 32
WIDTH = 32
#image size for data y
TARGET_HEIGHT = 64
TARGET_WIDTH =  64
#image depth when doing multiple image packing
DEPTH = 5

# package choice: akiyo_package; container_package; shepp_logan_phantom_package
(AMOUNT, x_set, y_set) = container_package(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, DEPTH)

# Separate training set and test set-----------------------------------------------
FRACTION = 0.8

(train_x, test_x) = separator('x', x_set, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)
(train_y, test_y) = separator('y', y_set, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)

# Build model------------------------------------------------------------------------
# model = super_resolution_model(TARGET_HEIGHT, TARGET_WIDTH)
model = SRnet_3d_model(AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH)

print('shape of train_x', train_x.shape, 'shape of train_y', train_y.shape)

# Train model ------------------------------------------------------------------------
EPOCHS = 500
BATCH = 20
VALIDATION_BATCH_SIZE = 20
STEPS_PER_EPOCH = math.floor(AMOUNT / BATCH)
VALIDATION_STEPS = math.floor(AMOUNT / VALIDATION_BATCH_SIZE)

history = model.fit(train_x, train_y, BATCH, EPOCHS, verbose=2, validation_split=1 - FRACTION)  #, callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=2, mode='auto', baseline=None, restore_best_weights=False)]

# https://keras.io/models/model/

#plot_history(history)

mean_squared_error, ssim_trained, psnr_trained = model.evaluate(train_x, train_y, verbose=1)

# Show performance---------------------------------------------------------------------
print(' ')

TARGET_HEIGHT_TMP = TARGET_HEIGHT - 2*2
TARGET_WIDTH_TMP = TARGET_WIDTH - 2*2

x = numpy.reshape(train_x[:, 2, 2:TARGET_HEIGHT_TMP + 2, 2:TARGET_WIDTH_TMP + 2, :], (math.floor(FRACTION*AMOUNT), TARGET_HEIGHT_TMP, TARGET_WIDTH_TMP))
y = numpy.reshape(train_y, (math.floor(FRACTION*AMOUNT), TARGET_HEIGHT_TMP, TARGET_WIDTH_TMP))

(psnr_before, ssim_before) = PSNR_SSIM(y,x)

# Make prediction---------------------------------------------------------------------
test_predictions = model.predict(test_x)
#print('type(test_predictions: ', type(test_predictions))
#print('test_predictions.shape: ', test_predictions.shape)

#print('after model.fit, train_x and train_y shape:', train_x.shape, train_y.shape)
#print('shape pf test_x', test_x.shape, 'shape of test_y', test_y.shape)

x = numpy.reshape(test_x[:, 2, 2:TARGET_HEIGHT_TMP+2, 2:TARGET_WIDTH_TMP+2, :], (AMOUNT - math.floor(FRACTION*AMOUNT), TARGET_HEIGHT_TMP, TARGET_WIDTH_TMP))
x_predict = numpy.reshape(test_predictions, (AMOUNT - math.floor(FRACTION*AMOUNT), TARGET_HEIGHT_TMP, TARGET_WIDTH_TMP))
y = numpy.reshape(test_y, (AMOUNT - math.floor(FRACTION*AMOUNT), TARGET_HEIGHT_TMP, TARGET_WIDTH_TMP))

(psnr_predict, ssim_predict) = PSNR_SSIM(y,x_predict)
(psnr_raw, ssim_raw) = PSNR_SSIM(y,x)

#compare accuracy using test_y
(mae_predict, mse_predict) = MAE_MSE(y, x_predict)
(mae_raw, mse_raw) = MAE_MSE(y, x)

mean_squared_error_test, ssim_test, psnr_test = model.evaluate(test_x, test_y, verbose=1)
print('mean_squared_error_test, ssim_test, psnr_test', mean_squared_error_test, ssim_test, psnr_test)

print('AMOUNT, BATCH, TARGET_HEIGHT:', AMOUNT, BATCH, TARGET_HEIGHT)

print('without processed by neural network:')
print('psnr = ', psnr_before)
print('ssim = ', ssim_before)

print('performance after processed by nerual network:')
print('psnr: ', psnr_trained)
print('ssim: ', ssim_trained)

print('performance of predicted images:')
print('psnr, ssim for raw data:', psnr_raw, ssim_raw)
print('psnr, ssim for predicted result:', psnr_predict, ssim_predict)
print('mae, mse for raw data:', mae_raw, mse_raw)
print('mae, mse for predicted result', mae_predict, mse_predict)


'''
fig=plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(test_predictions[0, :, :, 0])
fig.add_subplot(1,2,2)
plt.imshow(test_x[0, :, :, 0])
plt.show()
'''








