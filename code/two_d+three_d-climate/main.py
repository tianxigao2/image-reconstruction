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
AMOUNT = 2000
TOTAL_AMOUNT = 2000

# TODO:check acc.py
# max_val in acc.py is 255

#image depth when doing multiple image packing
DEPTH = 5
HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH = 159, 320, 318, 640 #default


# package choice: akiyo_package; container_package; shepp_logan_phantom_package
(x_set, y_set, TARGET_HEIGHT, TARGET_WIDTH, HEIGHT, WIDTH, AMOUNT, TOTAL_AMOUNT) = climate_2D(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)

print('x_set.shape, y_set.shape, TARGET_HEIGHT, TARGET_WIDTH, HEIGHT, WIDTH',
    x_set.shape, y_set.shape, TARGET_HEIGHT, TARGET_WIDTH, HEIGHT, WIDTH)

# Separate training set and test set-----------------------------------------------
FRACTION = 0.8

(train_x, test_x) = separator(x_set, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)
(train_y, test_y) = separator(y_set, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)

# Performance without any processing-------------------------------------------------
TARGET_HEIGHT_TMP = TARGET_HEIGHT - 2*2
TARGET_WIDTH_TMP = TARGET_WIDTH - 2*2

x = numpy.reshape(x_set[:, 2:TARGET_HEIGHT_TMP + 2, 2:TARGET_WIDTH_TMP + 2, :],(
    AMOUNT, TARGET_HEIGHT_TMP, TARGET_WIDTH_TMP))
y = numpy.reshape(y_set[:, 2:TARGET_HEIGHT_TMP + 2, 2:TARGET_WIDTH_TMP + 2, :],(
    AMOUNT, TARGET_HEIGHT_TMP, TARGET_WIDTH_TMP))

print('x.shape:', x.shape, 'y.shape', y.shape)

(psnr_before, ssim_before) = PSNR_SSIM(y,x)

# Build model------------------------------------------------------------------------
# choose a model to read in
# TODO: changed
#FILE_INDEX = 0
FILENAME = 'x3.mat'

model = transfer_learning_SRCNN(FILENAME, TARGET_HEIGHT, TARGET_WIDTH)

# Train model ------------------------------------------------------------------------
EPOCHS = 120
BATCH = 25
VALIDATION_BATCH_SIZE = 10
STEPS_PER_EPOCH = math.floor(AMOUNT / BATCH)
VALIDATION_STEPS = math.floor(AMOUNT / VALIDATION_BATCH_SIZE)

history = model.fit(train_x, train_y, BATCH, EPOCHS, verbose=2, validation_split=1 - FRACTION,
callbacks=[EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto', baseline=None, restore_best_weights=False)])
model.save(save_model_path() +'2d_model_container.h5')
print('save the first model')

# make prediction after individual processing ----------------------------------------
#train_prediction = model.predict(train_x)
#test_prediction = model.predict(test_x)
prediction = model.predict(x_set)
# y_set doesn't need any change

# Performance after 2d model process-------------------------------------------------
x_prediction = numpy.reshape(prediction[:, 2:TARGET_HEIGHT_TMP + 2, 2:TARGET_WIDTH_TMP + 2, :],(
    AMOUNT, TARGET_HEIGHT_TMP, TARGET_WIDTH_TMP))

(psnr_before_reload, ssim_before_reload) = PSNR_SSIM(y,x_prediction)

#=====================================================================================================

# store the predicted frames-----------------------------------------------------------
# append train_prediction with test_prediction
#interim = numpy.append(train_prediction, test_prediction, axis=0)
interim = prediction.reshape((-1, TARGET_HEIGHT, TARGET_WIDTH))
(TOTAL_AMOUNT, TARGET_HEIGHT, TARGET_WIDTH) = interim.shape

for i in range(TOTAL_AMOUNT):
    img = interim[i , : , :]
    img = Image.fromarray(img)
    img.save(prediction_path() + 'frame' + str(i) + '.tif')
    #print('save predicted frame' + str(i) + '.tif')

# Reload data-------------------------------------------------------------------------
(AMOUNT, x_set_reload, HEIGHT, WIDTH) = prediction_package(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, DEPTH)
print('prediction_set shape: ', x_set_reload.shape) #(296, 5, 352, 288, 1)

# Performance after reloading-------------------------------------------------
x_reload = numpy.reshape(x_set_reload[:, 2, 2:TARGET_HEIGHT_TMP + 2, 2:TARGET_WIDTH_TMP + 2, :],(
    AMOUNT, TARGET_HEIGHT_TMP, TARGET_WIDTH_TMP))
# y_set change picture size
predict_y_set = y_set[:, 2:TARGET_HEIGHT-2, 2:TARGET_WIDTH-2, :]
y_reload = numpy.reshape(predict_y_set, (AMOUNT, TARGET_HEIGHT_TMP, TARGET_WIDTH_TMP))

(psnr_after_reload, ssim_after_reload) = PSNR_SSIM(y_reload,x_reload)

#=====================================================================================================

# Separate training set and test set-----------------------------------------------
FRACTION = 0.9

#(train_prediction, test_prediction) = separator(x_set, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)
(train_x_reload, test_x_reload) = separator(x_set_reload, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)
(train_y_predict, test_y_predict) = separator(predict_y_set, FRACTION, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)

# Pass the first results to the 3D SRnet model ----------------------------------------
model = SRnet_3d_model(AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH)

# Train model ------------------------------------------------------------------------
# TODO: enlarge EPOCHS if time/memory permits
EPOCHS = 200
BATCH = 10
VALIDATION_BATCH_SIZE = 20
STEPS_PER_EPOCH = math.floor(AMOUNT / BATCH)
VALIDATION_STEPS = math.floor(AMOUNT / VALIDATION_BATCH_SIZE)

history = model.fit(train_x_reload, train_y_predict, BATCH, EPOCHS, verbose=2, validation_split=1 - FRACTION,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=2, mode='auto', baseline=None, restore_best_weights=False),
                               ModelCheckpoint(filepath = './checkpoint.h5', monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='auto', period=1)])
model.save(save_model_path() + '/3d_model_container.h5')
print('save the second model')

# Performance after 3d model process-------------------------------------------------
#mse_3d_train, ssim_3d_train, psnr_3d_train = model.evaluate(train_x_reload, train_y_predict, verbose=2)
#mse_3d_test, ssim_3d_test, psnr_3d_test = model.evaluate(test_x_reload, train_y_predict, verbose=2)

x_final_set = model.predict(x_set_reload)
x_final = numpy.reshape(x_final_set[:, 2:TARGET_HEIGHT_TMP + 2, 2:TARGET_WIDTH_TMP + 2, :],(
    AMOUNT, TARGET_HEIGHT_TMP, TARGET_WIDTH_TMP))

(psnr_final, ssim_final) = PSNR_SSIM(y_reload,x_final)


# Print performance-------------------------------------------------------------------
print('')
print('AMOUNT, BATCH, TARGET_HEIGHT, HEIGHT, TARGET_WIDTH, WIDTH:', AMOUNT, BATCH, TARGET_HEIGHT, HEIGHT, TARGET_WIDTH, WIDTH, '\n')

print('without any process, raw data:')
print('psnr_before, ssim_before:', psnr_before, ssim_before, '\n')

print('after SRCNN, before reloading:')
print('psnr_before_reload, ssim_before_reload:', psnr_before_reload, ssim_before_reload, '\n')

print('after reloading, before processing by 3d layers:')
print('psnr_after_reload, ssim_after_reload', psnr_after_reload, ssim_after_reload, '\n')

print('final result:')
print('psnr_final, ssim_final', psnr_final, ssim_final)

fig=plt.figure()
fig.add_subplot(2,2,1)
plt.imshow(x_set[0, :, :, 0])
plt.title('raw picture')
fig.add_subplot(2, 2, 2)
plt.imshow(prediction[0, :, :, 0])
plt.title('prediction before reloading')
fig.add_subplot(2,2,3)
plt.imshow(x_set_reload[0, :, :, 0])
plt.title('after reloading')
fig.add_subplot(2, 2, 4)
plt.imshow(x_final_set[0, :, :, 0])
plt.title('final prediction')
plt.show()
