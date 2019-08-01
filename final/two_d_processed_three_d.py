# -*- coding: utf-8 -*-
from load_data import *
from check_performance import *
from load_parameter import *
from fsrcnn import *
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

def two_d_processed_three_d(train_x, train_y, test_x, test_y, AMOUNT, AMOUNT_TEST, DEPTH, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):

    plt.imshow(test_x[0, :, :, 0])
    plt.title('raw test_x')
    plt.savefig('two_d_processed_three_d/raw_test.png')

    #no need to enlarge the train set

    #directly use trained model--------------------------------------------------------
    model_2d = keras.models.load_model('./FSRCNN/model_climate_fsrcnn.h5',
                    custom_objects={'ssim_for2d':ssim_for2d, 'psnr_for2d':psnr_for2d})

    # make prediction after individual processing ----------------------------------------
    interim_test = model_2d.predict(test_x)
    interim_train = model_2d.predict(train_x)

    plt.imshow(interim_test[0, :, :, 0])
    plt.title('test_x interim')
    plt.savefig('two_d_processed_three_d/interim_test_2d+3d.png')

    # Performance after 2d model process
    (psnr_before_pack, ssim_before_pack, mae_before_pack, mse_before_pack) = check_performance(interim_train, train_y, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT, without_padding = True)
    
    #modify dataset in shape---------------------------------------------------------------
    train_packed = repacking(interim_train)
    test_packed = repacking(interim_test)

    #cut the middle part of y_set out for ground-truth
    train_y_tmp = train_y[3:AMOUNT - 3, 2:TARGET_HEIGHT-2, 2:TARGET_WIDTH-2, :]
    train_y = train_y_tmp
    test_y_tmp = test_y[3:AMOUNT_TEST - 3, 2:TARGET_HEIGHT-2, 2:TARGET_WIDTH-2, :]
    test_y = test_y_tmp

    plt.imshow(test_packed[0, 2, :, :, 0])
    plt.title('test_x interim after packed')
    plt.savefig('two_d_processed_three_d/packed_interim_test_2d+3d.png')

    #Performance after data re-processing
    (psnr_packed, ssim_packed, mae_packed, mse_packed) = check_performance_3d(train_packed, train_y, TARGET_HEIGHT, TARGET_WIDTH, AMOUNT)

    model_3d = keras.models.load_model('./three_d_processed/3d_model_climate_300.h5',
                        custom_objects={'ssim_for2d':ssim_for2d, 'psnr_for2d':psnr_for2d})

    #make final prediction----------------------------------------------------------------------
    final = model_3d.predict(test_packed)
    train_final = model_3d.predict(train_packed)

    # Performance after 3d model process
    (psnr_final, ssim_final, mae_final, mse_final) = check_performance(train_final, train_y, TARGET_HEIGHT - 4, TARGET_WIDTH - 4, AMOUNT - 6, False)

    plt.imshow(final[0, :, :, 0])
    plt.title('test_x final result')
    plt.savefig('two_d_processed_three_d/final_test_2d+3d.png')
    plt.imshow(test_y[0, :, :, 0])
    plt.title('test_y')
    plt.savefig('two_d_processed_three_d/test_y_2d+3d.png')

    print('psnr_before_pack, ssim_before_pack, mae_before_pack, mse_before_pack')
    print(psnr_before_pack, ssim_before_pack, mae_before_pack, mse_before_pack, '\n')

    print('psnr_packed, ssim_packed, mae_packed, mse_packed')
    print(psnr_packed, ssim_packed, mae_packed, mse_packed,'\n')

    print('psnr_final, ssim_final, mae_final, mse_final')
    print(psnr_final, ssim_final, mae_final, mse_final, '\n')

    return

    

