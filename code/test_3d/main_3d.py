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
from keras.callbacks import EarlyStopping, ModelCheckpoint


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
HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH = 176, 144, 352, 288 #default
#image depth when doing multiple image packing
DEPTH = 5

# package choice: akiyo_package; container_package; shepp_logan_phantom_package
#(AMOUNT, x_set, y_set) = container_package(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, DEPTH)
(x_set, y_set, TARGET_HEIGHT, TARGET_WIDTH, HEIGHT, WIDTH) = container_2D(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
(AMOUNT, x_set, HEIGHT, WIDTH) = prediction_package(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, DEPTH)
print('prediction_set shape: ', x_set.shape)

AMOUNT = 296
TOTAL_AMOUNT = 296
# Separate training set and test set-----------------------------------------------
# y_set change picture size
predict_y_set = y_set[:, 2:TARGET_HEIGHT-2, 2:TARGET_WIDTH-2, :]
y_set = predict_y_set

# Separate training set and test set-----------------------------------------------
FRACTION = 0.9

(train_prediction, test_prediction) = separator(x_set, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)
(train_y, test_y) = separator(y_set, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)

# Build model------------------------------------------------------------------------
# model = super_resolution_model(TARGET_HEIGHT, TARGET_WIDTH)
model = SRnet_3d_model(AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH)

print('shape of train_prediction', train_prediction.shape, 'shape of train_y', train_y.shape)

# Train model ------------------------------------------------------------------------
EPOCHS = 300
BATCH = 10
VALIDATION_BATCH_SIZE = 20
STEPS_PER_EPOCH = math.floor(AMOUNT / BATCH)
VALIDATION_STEPS = math.floor(AMOUNT / VALIDATION_BATCH_SIZE)

#history = model.fit(train_x, train_y, BATCH, EPOCHS, verbose=2, validation_split=1 - FRACTION)  #, callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=2, mode='auto', baseline=None, restore_best_weights=False)]
history = model.fit(train_prediction, train_y, BATCH, EPOCHS, verbose=2, validation_split=1 - FRACTION,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=2, mode='auto', baseline=None, restore_best_weights=False),
                    ModelCheckpoint(filepath = './checkpoint.h5', monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='auto', period=1)])

# https://keras.io/models/model/

#plot_history(history)

mean_squared_error, ssim_trained, psnr_trained = model.evaluate(train_prediction, train_y, verbose=1)

# Show performance---------------------------------------------------------------------
print(' ')

TARGET_HEIGHT_TMP = TARGET_HEIGHT - 2*2
TARGET_WIDTH_TMP = TARGET_WIDTH - 2*2

x = numpy.reshape(train_prediction[:, 2, 2:TARGET_HEIGHT_TMP + 2, 2:TARGET_WIDTH_TMP + 2, :], (math.floor(FRACTION*AMOUNT), TARGET_HEIGHT_TMP, TARGET_WIDTH_TMP))
y = numpy.reshape(train_y, (math.floor(FRACTION*AMOUNT), TARGET_HEIGHT_TMP, TARGET_WIDTH_TMP))

(psnr_before, ssim_before) = PSNR_SSIM(y,x)

# Performance after 3d model process-------------------------------------------------
mse_3d_train, ssim_3d_train, psnr_3d_train = model.evaluate(train_prediction, train_y, verbose=2)
mse_3d_test, ssim_3d_test, psnr_3d_test = model.evaluate(test_prediction, test_y, verbose=2)

# Print performance-------------------------------------------------------------------
print('')
print('AMOUNT, BATCH, TARGET_HEIGHT, HEIGHT, TARGET_WIDTH, WIDTH:', AMOUNT, BATCH, TARGET_HEIGHT, HEIGHT, TARGET_WIDTH, WIDTH, '\n')

print('without any process, raw data:')
print('psnr_before, ssim_before:', psnr_before, ssim_before, '\n')

print('after processed by 3D SRnet:')
print('for train set, mse, ssim, psnr:', mse_3d_train, ssim_3d_train, psnr_3d_train)
print('for test set, mse, ssim, psnr:', mse_3d_test, ssim_3d_test, psnr_3d_test, '\n')




fig=plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(x_set[0, 2, :, :, 0])
fig.add_subplot(1,2,2)
plt.imshow(y_set[0, :, :, 0])
plt.show()









