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
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#---------------------------------------------------------------------------------#

#=================================================================================
'''
# use shepp-logan-phantom for testing---------------------------------------------
AMOUNT = 2000    #might be changed after picking frames
TOTAL_AMOUNT = 2000

# change max_val in acc.py to 1
'''
# use any other set for testing -------------------------------------------------------
AMOUNT = 2000   #might be changed after picking frames
TOTAL_AMOUNT = 2000
#image size for data x
(HEIGHT, WIDTH) = (159, 320)
#image size for data y
(TARGET_HEIGHT, TARGET_WIDTH) = (318, 640)
#image depth when doing multiple image packing
DEPTH = 5
#path to access the data
(x_set, y_set, TARGET_HEIGHT, TARGET_WIDTH, HEIGHT, WIDTH, AMOUNT, TOTAL_AMOUNT) = climate_2D(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)

# Separate training set and test set-----------------------------------------------
FRACTION = 0.8

(train_x, test_x) = separator('x', x_set, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)
(train_y, test_y) = separator('y', y_set, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)

# Build model------------------------------------------------------------------------
# model = super_resolution_model(TARGET_HEIGHT, TARGET_WIDTH)
model = SRnet_3d_model(AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH)

print('shape of train_x', train_x.shape, 'shape of train_y', train_y.shape)

# Train model ------------------------------------------------------------------------
EPOCHS = 300
BATCH = 20
VALIDATION_BATCH_SIZE = 20
STEPS_PER_EPOCH = math.floor(AMOUNT / BATCH)
VALIDATION_STEPS = math.floor(AMOUNT / VALIDATION_BATCH_SIZE)

history = model.fit(train_x, train_y, BATCH, EPOCHS, verbose=2, validation_split=1 - FRACTION,
    callbacks=[EarlyStopping(monitor='val_loss', patience=4, verbose=2, mode='auto', baseline=None, restore_best_weights=False)])

# https://keras.io/models/model/

#plot_history(history)

#mean_squared_error, ssim_trained, psnr_trained = model.evaluate(train_x, train_y, verbose=1)

# Show performance---------------------------------------------------------------------
print(' ')

# Performance without any processing-------------------------------------------------
TARGET_HEIGHT_TMP = TARGET_HEIGHT - 2*2
TARGET_WIDTH_TMP = TARGET_WIDTH - 2*2

x = numpy.reshape(test_x[:, 2, 2:TARGET_HEIGHT_TMP + 2, 2:TARGET_WIDTH_TMP + 2, :],(
    math.floor((1 - FRACTION) * AMOUNT), TARGET_HEIGHT_TMP, TARGET_WIDTH_TMP))
y = numpy.reshape(test_y[:, 2:TARGET_HEIGHT_TMP + 2, 2:TARGET_WIDTH_TMP + 2, :],(
    math.floor((1 - FRACTION) * AMOUNT), TARGET_HEIGHT_TMP, TARGET_WIDTH_TMP))

print('x.shape:', x.shape, 'y.shape', y.shape)

(psnr_raw, ssim_raw) = PSNR_SSIM(y,x)
(mae_raw, mse_raw) = MAE_MSE(y, x)

# Make prediction---------------------------------------------------------------------
test_predictions = model.predict(test_x)

x_predict = numpy.reshape(test_predictions[:, 2:TARGET_HEIGHT_TMP + 2, 2:TARGET_WIDTH_TMP + 2, :], (
    AMOUNT - math.floor(FRACTION*AMOUNT), TARGET_HEIGHT_TMP, TARGET_WIDTH_TMP))

(psnr_predict, ssim_predict) = PSNR_SSIM(y,x_predict)
(mae_predict, mse_predict) = MAE_MSE(y, x_predict)


print('without processed by neural network:')
print('psnr = ', psnr_raw)
print('ssim = ', ssim_raw)
print('mae = ', mae_raw)
print('mse = ', mse_raw, '\n')

print('performance after processed by nerual network:')
print('psnr: ', psnr_predict)
print('ssim: ', ssim_predict)
print('mae = ', mae_predict)
print('mse = ', mse_predict)


#fig=plt.figure()
#fig.add_subplot(1,3,1)
plt.imshow(test_x[0, :, :, 0])
plt.title('raw')
plt.savefig('raw.png')
#fig.add_subplot(1,3,2)
plt.imshow(test_predictions[0, :, :, 0])
plt.title('prediction')
plt.savefig('prediction.png')
#fig.add_subplot(1,3,3)
plt.imshow(test_y[0, :, :, 0])
plt.title('ground truth')
plt.savefig('groundtruth.png')
#plt.show()






