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

# definition parameters about dataset--------------------------------------------------------------------

# use Timofte data ========================================================================================
AMOUNT = 3288   #might be changed after picking frames
TOTAL_AMOUNT = 3288
#image size for data x
HEIGHT = 32
WIDTH = 32
#image size for data y#=====================================check where the BS print-out are from===============================================

TARGET_HEIGHT = 64
TARGET_WIDTH =  64
#image depth when doing multiple image packing
DEPTH = 5
#path to access the data
(x_set, y_set) = get_Timofte(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)

# Separate training set and test set-----------------------------------------------
FRACTION = 0.8

(train_x, test_x) = separator('x', x_set, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)
(train_y, test_y) = separator('y', y_set, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)

# Build model------------------------------------------------------------------------
model = super_resolution_model(TARGET_HEIGHT, TARGET_WIDTH)

# Train model ------------------------------------------------------------------------
EPOCHS = 1000
BATCH = 100
VALIDATION_BATCH_SIZE = 10
STEPS_PER_EPOCH = math.floor(AMOUNT / BATCH)
VALIDATION_STEPS = math.floor(AMOUNT / VALIDATION_BATCH_SIZE)

history = model.fit(train_x, train_y, BATCH, EPOCHS, verbose=2, validation_split=1 - FRACTION, callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=False)])
# https://keras.io/models/model/

#plot_history(history)

mean_squared_error, ssim_trained, psnr_trained = model.evaluate(train_x, train_y, verbose=2)

# Show performance---------------------------------------------------------------------
print(' ')

x = numpy.reshape(train_x, (math.floor(FRACTION*AMOUNT), TARGET_HEIGHT, TARGET_WIDTH))
y = numpy.reshape(train_y, (math.floor(FRACTION*AMOUNT), TARGET_HEIGHT, TARGET_WIDTH))

psnr_before = PSNR(y, x)
ssim_before = SSIM(y, x)

# Make prediction---------------------------------------------------------------------
test_predictions = model.predict(test_x)
#print(type(test_predictions))
# Float32, (20, 64, 64, 1)
x = numpy.reshape(test_x, (AMOUNT - math.floor(FRACTION*AMOUNT), TARGET_HEIGHT, TARGET_WIDTH))
x_predict = numpy.reshape(test_predictions, (AMOUNT - math.floor(FRACTION*AMOUNT), TARGET_HEIGHT, TARGET_WIDTH))
y = numpy.reshape(test_y, (AMOUNT - math.floor(FRACTION*AMOUNT), TARGET_HEIGHT, TARGET_WIDTH))
print('finish prediction')

psnr_predict = PSNR(y, x_predict)
ssim_predict = SSIM(y, x_predict)
(psnr_raw, ssim_raw) = (PSNR(y, x), SSIM(y, x))

#compare accuracy using test_y
(mae_predict, mse_predict) = MAE_MSE(y, x_predict)
(mae_raw, mse_raw) = MAE_MSE(y, x)

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








