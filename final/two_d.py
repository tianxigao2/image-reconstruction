# -*- coding: utf-8 -*-
from load_data import *
from load_parameter import *
from check_performance import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv3D
from keras import optimizers
from keras.optimizers import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import math
import numpy
import matplotlib.pyplot as plt

def transfer_learning_SRCNN(filename, HEIGHT, WIDTH):

    par = load_parameter(filename)

    model = Sequential()
    
    model.add(Conv2D(64, kernel_size = (9,9), padding = 'same', activation = 'relu', use_bias = True,
    input_shape=(HEIGHT,WIDTH,1), trainable = False))
    model.add(Conv2D(32, (1, 1), padding='same', activation='relu', use_bias = True, trainable = True))
    model.add(Conv2D(1, (5, 5), padding='same', activation = 'relu', use_bias = True, trainable = True))

    for i in range(3):
        model.layers[i].set_weights(par[i])

    print(model.summary())

    model.compile(loss = 'mean_squared_error', optimizer = adam(lr=0.0005, decay=1e-6), metrics=[ssim_for2d, psnr_for2d])

    return model

def two_d(train_x, train_y, test_x, test_y, AMOUNT, AMOUNT_TEST, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):
    plt.imshow(test_x[0, :, :, 0])
    plt.title('raw test_x')
    plt.savefig('two_d/raw_test_2d.png')

    #enlarge train set x for neural network--------------------------------------------------
    train_x = enlarge(train_x, TARGET_HEIGHT, TARGET_WIDTH)
    test_x = enlarge(test_x, TARGET_HEIGHT, TARGET_WIDTH)

    #plt.imshow(test_x[0, :, :, 0])
    #plt.show()

    #check performance for raw data (only processed by bicubic)
    (psnr_raw, ssim_raw, mae_raw, mse_raw) = check_performance(train_x, train_y, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT, without_padding = False)

    #load the 2d model--------------------------------------------------------------------------
    transfer_learning_FILE = '/home/user1/REUS/image-reconstruction-2019-a/papers_and_codes/SRCNN/SRCNN_test/model/9-1-5(ImageNet)/x3.mat' #the fine-tuned-parameter file for transfer learning model
    model_2d = transfer_learning_SRCNN(transfer_learning_FILE, TARGET_HEIGHT, TARGET_WIDTH)

    # Train the 2d model ------------------------------------------------------------------------
    EPOCHS = 120
    BATCH = 25
    FRACTION = 0.9
    
    model_2d.fit(train_x, train_y, BATCH, EPOCHS, verbose=2, validation_split=1 - FRACTION, shuffle = True,
    callbacks=[EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto', baseline=None, restore_best_weights=False),
               ModelCheckpoint(filepath = './two_d/2d_model_climate_300.h5', monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='auto', period=1)])

    model_2d = keras.models.load_model('./two_d/2d_model_climate_300.h5',
                    custom_objects={'ssim_for2d':ssim_for2d, 'psnr_for2d':psnr_for2d})
                    
    # make prediction after individual processing ----------------------------------------
    final_test = model_2d.predict(test_x)
    final_train = model_2d.predict(train_x)

    # Performance after 2d model process
    (psnr, ssim, mae, mse) = check_performance(final_train, train_y, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT, without_padding = False)

    plt.imshow(final_test[0, :, :, 0])
    plt.title('test_x final result')
    plt.savefig('two_d/final_test_2d.png')
    plt.imshow(test_y[0, :, :, 0])
    plt.title('test_y')
    plt.savefig('two_d/test_y_2d.png')

    print('psnr_raw, ssim_raw, mae_raw, mse_raw: ')
    print(psnr_raw, ssim_raw, mae_raw, mse_raw, '\n')

    print('psnr_final, ssim_final, mae_final, mse_final')
    print(psnr, ssim, mae, mse, '\n')

    return
