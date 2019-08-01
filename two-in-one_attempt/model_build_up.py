from acc import *
from load_parameter import *
from load_data import *
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Conv3D, Reshape
from keras import optimizers
from keras.optimizers import adam
from keras import backend
import tensorflow as tf
import numpy
import math

#TODO: depth defaults to be 5; If modified, need to change lambda function and lambda shape function

def base_model_SRCNN(FILENAME, HEIGHT, WIDTH):

    par = load_parameter(FILENAME)  #load transfer learning parameters
    
    model = Sequential()
    
    model.add(Conv2D(64, (9,9), padding = 'same', activation ='relu', use_bias = True,
    input_shape=(HEIGHT,WIDTH, 1), trainable = False))
    model.add(Conv2D(32, (1, 1), padding='same', activation='relu', use_bias = True, trainable = True))
    model.add(Conv2D(1, (5, 5), padding='same', activation = 'relu', use_bias = True, trainable = True))

    for i in range(3):
        model.layers[i].set_weights(par[i])

    #print(model.summary())
    #model.compile(loss = 'mean_squared_error', optimizer = adam(lr=0.0005, decay=1e-6), metrics=[ssim_for2d, psnr_for2d])

    return model

def repacking(x):
    #----------------------------get shape from input tensor---------------------------------
    (AMOUNT, TARGET_HEIGHT, TARGET_WIDTH, tmp) = x.shape
    AMOUNT = keras.backend.int_shape(x)[0]#list(x.shape)[0]
    print('shape of AMOUNT', AMOUNT)
    DEPTH = 5
    #AMOUNT = 26
    #----------------------------pack frame one by one---------------------------------------
    FIRST = True
    HALF_RANGE = math.floor(DEPTH/2)

    for i in range(AMOUNT):
        if (i - HALF_RANGE) < 0 or (i + HALF_RANGE) >= AMOUNT:
            AMOUNT = AMOUNT - 1
        else:
            if DEPTH%2 == 0:
                RANGE = range(i - HALF_RANGE, i + HALF_RANGE)
            else:
                RANGE = range(i - HALF_RANGE, i + HALF_RANGE + 1)

            for j in RANGE:
                frame = x[j, :, :, :]  #(352, 288, 1), type = tensor
                frame = tf.reshape(frame,(1, TARGET_HEIGHT, TARGET_WIDTH))
                
                if j == i - HALF_RANGE:
                    package = frame
                else:   
                    package = tf.concat([package, frame], 0)

            if FIRST == True:
                package_set = package
                FIRST = False
            else:
                package_set = tf.concat([package_set, package], 0)

    package_set = tf.reshape(package_set, (AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH, 1))   #(294, 5, 352, 288, 1)
    return package_set
    


def repacking_output_shape(input_shape):
    #--------------get shape from input_shape---------------------------
    [AMOUNT, TARGET_HEIGHT, TARGET_WIDTH, tmp] = list(input_shape)
    DEPTH = 5
    #AMOUNT = 26
    #--------------modify shape-----------------------------------------
    HALF_RANGE = math.floor(DEPTH/2)
    for i in range(AMOUNT):
        if (i - HALF_RANGE) < 0 or (i + HALF_RANGE) >= AMOUNT:
            AMOUNT = AMOUNT - 1
    
    output_shape = (AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH, 1)
    return output_shape

def combined(FILENAME, AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH):
    #----------------------------build SRCNN model-----------------------------------------   
    ip = Input(shape = (TARGET_HEIGHT, TARGET_WIDTH, 1))
    SRCNN_network = base_model_SRCNN(FILENAME, TARGET_HEIGHT, TARGET_WIDTH) (ip)     
    
    #----------------------------pack frame one by one---------------------------------------
    interim_repacking = Lambda(repacking, output_shape = repacking_output_shape)(SRCNN_network)

    #---------------------------build 3dSRnet model--------------------------------------------
    SRnet1 = Conv3D(32, kernel_size = (3, 3, 3), padding='same', data_format="channels_last",
                    activation='relu', use_bias=True)(interim_repacking)   
                    #input_shape=(DEPTH, TARGET_HEIGHT, TARGET_WIDTH,1)
    SRnet2 = Conv3D(32, kernel_size = (3, 3, 3), strides=(1, 1, 1), padding='same', dilation_rate=(1, 1, 1), activation='relu',
                    use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros') (SRnet1)
    SRnet3 = Conv3D(32, kernel_size = (3, 3, 3), strides=(1, 1, 1), padding='same', dilation_rate=(1, 1, 1), activation='relu',
                    use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros') (SRnet2)
    SRnet_downgrade_1 = Conv3D(32, kernel_size = (3, 3, 3), strides=(1, 1, 1), padding='valid', dilation_rate=(1, 1, 1), activation='relu',
                    use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros') (SRnet3)
    SRnet_downgrade_2 = Conv3D(32, kernel_size = (3, 3, 3), strides=(1, 1, 1), padding='valid', dilation_rate=(1, 1, 1), activation='relu',
                    use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros') (SRnet_downgrade_1)
    SRnet_reshape = Reshape((TARGET_HEIGHT - 2*2, TARGET_WIDTH - 2*2, -1))(SRnet_downgrade_2)
    SRnet_output = Conv2D(1, kernel_size=(3, 3), padding='same', activation='relu', use_bias = True,
                    input_shape=(DEPTH,TARGET_HEIGHT-2*2, TARGET_WIDTH-2*2, -1))(SRnet_reshape)
    
    #--------------------------test the result of combination----------------------------------
    combined_model = Model(inputs = ip, outputs = SRnet_output)
    print(combined_model.summary())  

    return combined_model  
