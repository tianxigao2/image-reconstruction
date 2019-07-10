from acc import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv3D
from keras import optimizers
from keras.optimizers import adam

def super_resolution_model(HEIGHT, WIDTH):
    
    model = Sequential()
    
    model.add(Conv2D(64, kernel_size=(9, 9), padding='same', activation='relu', use_bias = True,  input_shape=(HEIGHT,WIDTH,1)))
    #model.add(Conv2D(32, (1, 1), padding = 'same', activation = 'relu', use_bias = True))
    model.add(Conv2D(32, (1, 1), padding='same', activation='relu', use_bias = True))
    model.add(Conv2D(1, (5, 5), padding='same', activation = 'relu', use_bias = True))
    
    print(model.summary())

    '''
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 64, 64, 64)        5248      
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 64, 64, 32)        2080      
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 64, 64, 1)         801       
    =================================================================
    Total params: 8,129
    Trainable params: 8,129
    Non-trainable params: 0
    '''

    model.compile(loss = 'mean_squared_error', optimizer = adam(lr=0.001, decay=1e-6), metrics=[ssim_for2d, psnr_for2d])

    return model
