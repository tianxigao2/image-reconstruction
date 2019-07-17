from acc import *
from load_parameter import *
from load_data import *
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv3D, Reshape
from keras import optimizers
from keras.optimizers import adam
from keras import backend
import tensorflow as tf
import numpy
import math

def base_model_SRCNN(FILENAME, HEIGHT, WIDTH):

    par = load_parameter(FILENAME)  #load transfer learning parameters
    
    model = Sequential()
    
    model.add(Conv2D(64, (9,9), padding = 'same', activation ='relu', use_bias = True,
    input_shape=(HEIGHT,WIDTH, 1), trainable = False))
    model.add(Conv2D(32, (1, 1), padding='same', activation='relu', use_bias = True, trainable = True))
    model.add(Conv2D(1, (5, 5), padding='same', activation = 'relu', use_bias = True, trainable = True))

    '''
    SRCNN1 = Conv2D(64, 9, padding = 'same', activation ='relu', use_bias = True, 
        data_format='channels_last', trainable = False) (ip)  #input_shape=(TARGET_HEIGHT, TARGET_WIDTH, 1),
    SRCNN2 = Conv2D(32, 1, padding='same', activation='relu', use_bias = True, trainable = True) (SRCNN1)
    SRCNN3 = Conv2D(1, 5, padding='same', activation = 'relu', use_bias = True, trainable = True) (SRCNN2)

    SRCNN = Model(inputs = ip, outputs = SRCNN3)
    print(SRCNN.summary())
    ''' 

    for i in range(3):
        model.layers[i].set_weights(par[i])

    print(model.summary())

    model.compile(loss = 'mean_squared_error', optimizer = adam(lr=0.0005, decay=1e-6), metrics=[ssim_for2d, psnr_for2d])

    return model

def SRnet_3d_model(AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH):
    '''
    SRnet1 = Conv3D(32, kernel_size = (3, 3, 3), padding='same', data_format="channels_last",
                    activation='relu', use_bias=True)(package_set_tensor)   #input_shape=(DEPTH, TARGET_HEIGHT, TARGET_WIDTH,1)
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
    '''
    model = Sequential()
    
    model.add(Conv3D(32, kernel_size = (3, 3, 3), padding='same', data_format="channels_last", activation='relu', use_bias=True, input_shape=(DEPTH, TARGET_HEIGHT, TARGET_WIDTH,1)))
    model.add(Conv3D(32, kernel_size = (3, 3, 3), strides=(1, 1, 1), padding='same', dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Conv3D(32, kernel_size = (3, 3, 3), strides=(1, 1, 1), padding='same', dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Conv3D(32, kernel_size = (3, 3, 3), strides=(1, 1, 1), padding='valid', dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Conv3D(32, kernel_size = (3, 3, 3), strides=(1, 1, 1), padding='valid', dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(Reshape((TARGET_HEIGHT - 2*2, TARGET_WIDTH - 2*2, -1)))    # first 2 for 2 lines of erosion; second 2 for 2 layers of erosion
    model.add(Conv2D(1, kernel_size=(3, 3), padding='same', activation='relu', use_bias = True, input_shape=(DEPTH,TARGET_HEIGHT - 2*2, TARGET_WIDTH - 2*2, -1)))
    
    print(model.summary())

    model.compile(loss = 'mean_squared_error', optimizer = adam(lr=0.002, decay=1e-5), metrics=[ssim_for2d, psnr_for2d])

    return model

def combined(FILENAME, AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH):
    #----------------------------build SRCNN model-----------------------------------------   
    ip = Input(shape = (TARGET_HEIGHT, TARGET_WIDTH, 1))
    SRCNN_network = base_model_SRCNN(FILENAME, TARGET_HEIGHT, TARGET_WIDTH) (ip)     

    #----------------------------pack frame one by one---------------------------------------
    FIRST = True
    HALF_RANGE = math.floor(DEPTH/2)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    interim_tensor = sess.run(SRCNN_network, feed_dict = {
        "input_1: 0": numpy.zeros((AMOUNT, TARGET_HEIGHT, TARGET_WIDTH, 1)).astype(numpy.float32)})

    for i in range(AMOUNT):
        if (i - HALF_RANGE) < 0 or (i + HALF_RANGE) >= AMOUNT:
            AMOUNT = AMOUNT - 1
        else:
            if DEPTH%2 == 0:
                RANGE = range(i - HALF_RANGE, i + HALF_RANGE)
            else:
                RANGE = range(i - HALF_RANGE, i + HALF_RANGE + 1)

            for j in RANGE:
                frame = interim_tensor[j, :, :, :]  #(352, 288, 1)
                
                frame = numpy.reshape(frame, (1, TARGET_HEIGHT, TARGET_WIDTH))  #(1, 352, 288)
                
                if j == i - HALF_RANGE:
                    package = frame
                else:              
                    package = numpy.append(package, frame, axis = 0)

            if FIRST == True:
                package_set = package
                FIRST = False
            else:
                package_set= numpy.append(package_set, package, axis = 0)

    package_set = numpy.reshape(package_set, (AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH, 1))   #(294, 5, 352, 288, 1)
    package_set_tensor= tf.convert_to_tensor(package_set, dtype = tf.float32)

    #---------------------------build 3dSRnet model--------------------------------------------
    '''
    #for testing only
    interim_ip = Input(shape = (DEPTH, TARGET_HEIGHT, TARGET_WIDTH,1))
    SRnet_layer_tmp = SRnet_3d_model(AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH)(interim_ip)
    SRnet_tmp = Model(inputs = interim_ip, outputs = SRnet_layer_tmp)
    print(SRnet_tmp.summary())
    '''
    #for combined model sequence
    SRnet_layer = SRnet_3d_model(AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH)(package_set_tensor)
    
    #--------------------------test the result of combination----------------------------------
    combined_model = Model(inputs = ip, outputs = SRnet_layer)
    print(combined_model.summary())  

    return combined_model  
