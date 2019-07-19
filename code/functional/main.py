from load_data import *
from separator import *
from model_build_up import *
from train import *
from getpath import *
from acc import *
from load_parameter import *
import numpy
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

# TODO: match the amount with dataset
# TODO: check acc.py
# TODO: enlarge EPOCHS if time/memory permits
# TODO: enlarge model.fit patience
# TODO: change filename for parameter loading

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
# TODO: match the amount with dataset
AMOUNT = 300
TOTAL_AMOUNT = 300

# TODO:check acc.py
# max_val in acc.py is 255

#image depth when doing multiple image packing
DEPTH = 5
HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH = 176, 144, 352, 288 #default


# package choice: akiyo_package; container_package; shepp_logan_phantom_package
(x_set, y_set, TARGET_HEIGHT, TARGET_WIDTH, HEIGHT, WIDTH) = container_2D(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
y_set_tmp = y_set[:, 2:TARGET_HEIGHT - 2, 2:TARGET_WIDTH - 2, :]
y_set = y_set_tmp

print('x_set.shape, y_set.shape, TARGET_HEIGHT, TARGET_WIDTH, HEIGHT, WIDTH')
print(x_set.shape, y_set.shape, TARGET_HEIGHT, TARGET_WIDTH, HEIGHT, WIDTH)


'''
# Performance without any processing-------------------------------------------------
TARGET_HEIGHT_TMP = TARGET_HEIGHT - 2*2
TARGET_WIDTH_TMP = TARGET_WIDTH - 2*2
x = numpy.reshape(train_x[:, 2:TARGET_HEIGHT_TMP + 2, 2:TARGET_WIDTH_TMP + 2, :], (math.floor(FRACTION*AMOUNT), TARGET_HEIGHT_TMP, TARGET_WIDTH_TMP))
y = numpy.reshape(train_y[:, 2:TARGET_HEIGHT_TMP + 2, 2:TARGET_WIDTH_TMP + 2, :], (math.floor(FRACTION*AMOUNT), TARGET_HEIGHT_TMP, TARGET_WIDTH_TMP))
print('x.shape:', x.shape, 'y.shape', y.shape)
(psnr_before, ssim_before) = PSNR_SSIM(y,x)
'''
# Build model------------------------------------------------------------------------
# choose a model to read in
# TODO: changed
#FILE_INDEX = 0
FILENAME = 'x3.mat'

model = combined(FILENAME, AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH)
model.compile(loss = 'mean_squared_error',
                optimizer = adam(lr=0.0005, decay=1e-6), metrics=[ssim_for2d, psnr_for2d])

# Train model ------------------------------------------------------------------------
EPOCHS = 120
BATCH = 25
VALIDATION_FRACTION = 0.8
VALIDATION_BATCH_SIZE = 10
STEPS_PER_EPOCH = math.floor(AMOUNT / BATCH)
VALIDATION_STEPS = math.floor(AMOUNT / VALIDATION_BATCH_SIZE)

history = model.fit(x_set, y_set, BATCH, EPOCHS, verbose=2, validation_split=1 - VALIDATION_FRACTION,
        callbacks=[EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto'),
        ModelCheckpoint(filepath = './combined_network.h5', monitor='val_loss', verbose=2,
                        save_best_only=True, save_weights_only=False, mode='auto', period=1)])

'''
# Performance after 2d model process-------------------------------------------------
mse_individual_processing_train, ssim_individual_processing_train, psnr_individual_processing_train = model.evaluate(train_x, train_y, verbose=2)
mse_individual_processing_test, ssim_individual_processing_test, psnr_individual_processing_test = model.evaluate(test_x, test_y, verbose=2)

# make prediction after individual processing ----------------------------------------
train_prediction = model.predict(train_x)
test_prediction = model.predict(test_x)
# y_set doesn't need any change

# store the predicted frames-----------------------------------------------------------
# append train_prediction with test_prediction
interim = numpy.append(train_prediction, test_prediction, axis=0)
interim = interim.reshape((-1, TARGET_HEIGHT, TARGET_WIDTH))
(TOTAL_AMOUNT, TARGET_HEIGHT, TARGET_WIDTH) = interim.shape

for i in range(TOTAL_AMOUNT):
    img = interim[i , : , :]
    img = Image.fromarray(img)
    img.save(prediction_path() + 'frame' + str(i) + '.tif')
    #print('save predicted frame' + str(i) + '.tif')

# Reload data-------------------------------------------------------------------------
(AMOUNT, x_set, HEIGHT, WIDTH) = prediction_package(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, DEPTH)

if AMOUNT != 296:
    AMOUNT = 296
    TOTAL_AMOUNT = 296

# y_set change picture size
predict_y_set = y_set[:, 2:TARGET_HEIGHT-2, 2:TARGET_WIDTH-2, :]
y_set = predict_y_set

# shape of train_prediction (266, 5, 352, 288, 1) shape of train_y (266, 348, 284, 1)

# Separate training set and test set-----------------------------------------------
FRACTION = 0.9

(train_prediction, test_prediction) = separator(x_set, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)
(train_y, test_y) = separator(y_set, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)

# Pass the first results to the 3D SRnet model ----------------------------------------
model = SRnet_3d_model(AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH)

# Train model ------------------------------------------------------------------------
# TODO: enlarge EPOCHS if time/memory permits
EPOCHS = 200
BATCH = 10
VALIDATION_BATCH_SIZE = 20
STEPS_PER_EPOCH = math.floor(AMOUNT / BATCH)
VALIDATION_STEPS = math.floor(AMOUNT / VALIDATION_BATCH_SIZE)

history = model.fit(train_prediction, train_y, BATCH, EPOCHS, verbose=2, validation_split=1 - FRACTION,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=2, mode='auto', baseline=None, restore_best_weights=False),
                               ModelCheckpoint(filepath = './checkpoint.h5', monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='auto', period=1)])
model.save(save_model_path()+'3d_model.h5')
print('save the second model')

# Performance after 3d model process-------------------------------------------------
mse_3d_train, ssim_3d_train, psnr_3d_train = model.evaluate(train_prediction, train_y, verbose=2)
mse_3d_test, ssim_3d_test, psnr_3d_test = model.evaluate(test_prediction, test_y, verbose=2)

# Print performance-------------------------------------------------------------------
print('')
print('AMOUNT, BATCH, TARGET_HEIGHT, HEIGHT, TARGET_WIDTH, WIDTH:', AMOUNT, BATCH, TARGET_HEIGHT, HEIGHT, TARGET_WIDTH, WIDTH, '\n')

print('without any process, raw data:')
print('psnr_before, ssim_before:', psnr_before, ssim_before, '\n')

print('after processed by SRCNN individually:')
print('for train set, mse, ssim, psnr:', mse_individual_processing_train, ssim_individual_processing_train, psnr_individual_processing_train)
print('for test set, mse, ssim, psnr:', mse_individual_processing_test, ssim_individual_processing_test, psnr_individual_processing_test, '\n')

print('after processed by 3D SRnet:')
print('for train set, mse, ssim, psnr:', mse_3d_train, ssim_3d_train, psnr_3d_train)
print('for test set, mse, ssim, psnr:', mse_3d_test, ssim_3d_test, psnr_3d_test, '\n')

'''



