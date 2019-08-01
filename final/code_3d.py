# -*- coding: utf-8 -*-
from load_data import *
from check_performance import *
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv3D
from keras import optimizers
from keras.optimizers import adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import math
import numpy
import matplotlib.pyplot as plt

def SRnet_3d_model(AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH):
    model = Sequential()

    model.add(Conv3D(32, kernel_size = (5, 5, 3), padding='same', data_format="channels_last", activation = 'relu', use_bias=True, input_shape=(DEPTH, TARGET_HEIGHT, TARGET_WIDTH,1)))
    
    model.add(Conv3D(8, kernel_size = (1, 1, 3), strides=(1, 1, 1), padding='same', activation = 'relu', dilation_rate=(1, 1, 1), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    
    model.add(Conv3D(8, kernel_size = (3, 3, 3), strides=(1, 1, 1), padding='same', activation = 'relu', dilation_rate=(1, 1, 1), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    
    model.add(Conv3D(8, kernel_size = (3, 3, 3), strides=(1, 1, 1), padding='same', activation = 'relu', dilation_rate=(1, 1, 1), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    
    model.add(Conv3D(32, kernel_size = (1, 1, 3), strides=(1, 1, 1), padding='same', activation = 'relu', dilation_rate=(1, 1, 1), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    
    model.add(Conv3D(32, kernel_size = (3, 3, 3), strides=(1, 1, 1), padding='valid', activation = 'relu', dilation_rate=(1, 1, 1), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    
    model.add(Conv3D(32, kernel_size = (3, 3, 3), strides=(1, 1, 1), padding='valid', activation = 'relu', dilation_rate=(1, 1, 1), use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    
    model.add(Reshape((TARGET_HEIGHT - 2*2, TARGET_WIDTH - 2*2, -1)))    # first 2 for 2 lines of erosion; second 2 for 2 layers of erosion
    model.add(Conv2D(1, kernel_size=(5, 5), padding='same', use_bias = True, input_shape=(DEPTH,TARGET_HEIGHT - 2*2, TARGET_WIDTH - 2*2, -1)))
    
    print(model.summary())
    model.compile(loss = 'mean_squared_error', optimizer = adam(lr=0.001, decay=1e-3), metrics=[ssim_for2d, psnr_for2d])

    return model

def three_d(train_x, train_y, test_x, test_y, AMOUNT, AMOUNT_TEST, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):
    
    plt.imshow(test_x[0, :, :, 0])
    plt.title('raw test_x')
    plt.savefig('three_d/raw_test_3d.png')

    #enlarge train set x for neural network--------------------------------------------------
    train_x = enlarge(train_x, TARGET_HEIGHT, TARGET_WIDTH)
    test_x = enlarge(test_x, TARGET_HEIGHT, TARGET_WIDTH)

    plt.imshow(test_x[0, :, :, 0])
    plt.title('enlarged test_x')
    plt.savefig('three_d/raw_test_3d.png')

    print('dtype of train_x:', train_x.dtype)

    #check performance for raw data (only processed by bicubic)
    (psnr_raw, ssim_raw, mae_raw, mse_raw) = check_performance(train_x, train_y, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT, without_padding = True)
    print('psnr_raw, ssim_raw, mae_raw, mse_raw: ')
    print(psnr_raw, ssim_raw, mae_raw, mse_raw, '\n')

    #cut the middle part of y_set out for ground-truth
    train_y_tmp = train_y[3:AMOUNT - 3, 2:TARGET_HEIGHT-2, 2:TARGET_WIDTH-2, :]
    train_y = train_y_tmp
    test_y_tmp = test_y[3:AMOUNT_TEST - 3, 2:TARGET_HEIGHT-2, 2:TARGET_WIDTH-2, :]
    test_y = test_y_tmp

    train_packed = repacking(train_x)   #train_packed:(114, 5, 318, 640, 1), train_y:(114, 318, 640, 1)
    test_packed = repacking(test_x) 
    
    plt.imshow(test_packed[0, 2, :, :, 0])
    plt.title('test_x after packed')
    plt.savefig('three_d/packed_test_3d.png')
    
    #Performance after data re-processing
    (psnr_packed, ssim_packed, mae_packed, mse_packed) = check_performance_3d(train_packed, train_y, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)
    print('psnr_packed, ssim_packed, mae_packed, mse_packed')
    print(psnr_packed, ssim_packed, mae_packed, mse_packed,'\n')

    #load the 3d model----------------------------------------------------------------------
    model_3d = SRnet_3d_model(AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH)

    # train the 3d model-----------------------------------------------------------------------
    EPOCHS = 100
    BATCH = 10
    FRACTION = 0.8

    model_3d.fit(train_packed, train_y, BATCH, EPOCHS, verbose=2, validation_split=1 - FRACTION,
    callbacks=[EarlyStopping(monitor='val_loss', patience=8, verbose=2, mode='auto', baseline=None, restore_best_weights=False),
               ModelCheckpoint(filepath = './three_d/3d_model_climate_300.h5', monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='auto', period=1)])

    model_3d = keras.models.load_model('./three_d/3d_model_climate_300.h5',
                        custom_objects={'ssim_for2d':ssim_for2d, 'psnr_for2d':psnr_for2d})
    
    #make final prediction----------------------------------------------------------------------
    final = model_3d.predict(test_packed)
    train_final = model_3d.predict(train_packed)

    plt.imshow(final[0, :, :, 0])
    plt.title('test_x final result')
    plt.savefig('three_d/final_test_3d.png')
    plt.imshow(test_y[0, :, :, 0])
    plt.title('test_y')
    plt.savefig('three_d/test_y_3d.png')

    # Performance after 3d model process
    (psnr_final, ssim_final, mae_final, mse_final) = check_performance(train_final, train_y, TARGET_HEIGHT - 4, TARGET_WIDTH - 4, AMOUNT - 6, False)

    print('psnr_raw, ssim_raw, mae_raw, mse_raw: ')
    print(psnr_raw, ssim_raw, mae_raw, mse_raw, '\n')

    print('psnr_packed, ssim_packed, mae_packed, mse_packed')
    print(psnr_packed, ssim_packed, mae_packed, mse_packed,'\n')

    print('psnr_final, ssim_final, mae_final, mse_final')
    print(psnr_final, ssim_final, mae_final, mse_final, '\n')

    return
