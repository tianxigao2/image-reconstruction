from acc import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv3D
from keras import optimizers
from keras.optimizers import adam

def SRnet_3d_model(AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH):

    '''
    The first 3D convolution layer takes a series of five consecutive frames in a sliding input window,
    where each 3D filter generates a temporal feature map (TFM).
    For an input of five frames and a 3D filter of depth 3,
    no extrapolation is carried out from layer L-1, where L is the number of convolution layers.
    '''
    
    model = Sequential()
    
    model.add(Conv3D(32, kernel_size = (3, 3, 3), padding='same', data_format="channels_last", activation='relu', use_bias=True, input_shape=(DEPTH, TARGET_HEIGHT, TARGET_WIDTH,1)))
    model.add(Conv3D(32, kernel_size = (3, 3, 3), strides=(1, 1, 1), padding='same', dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    model.add(Conv3D(32, kernel_size = (3, 3, 3), strides=(1, 1, 1), padding='same', dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    
    #add 2 layer
    model.add(Conv3D(32, kernel_size = (3, 3, 3), strides=(1, 1, 1), padding='same', dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    model.add(Conv3D(32, kernel_size = (3, 3, 3), strides=(1, 1, 1), padding='same', dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    
    model.add(Conv3D(32, kernel_size = (3, 3, 3), strides=(1, 1, 1), padding='valid', dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    model.add(Conv3D(32, kernel_size = (3, 3, 3), strides=(1, 1, 1), padding='valid', dilation_rate=(1, 1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    model.add(Reshape((TARGET_HEIGHT - 2*2, TARGET_WIDTH - 2*2, -1)))    # first 2 for 2 lines of erosion; second 2 for 2 layers of erosion
    model.add(Conv2D(1, kernel_size=(3, 3), padding='same', activation='relu', use_bias = True, input_shape=(DEPTH,TARGET_HEIGHT - 2*2, TARGET_WIDTH - 2*2, -1)))
    
    print(model.summary())

    '''
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv3d_1 (Conv3D)            (None, 5, 64, 64, 32)     896       
    _________________________________________________________________
    conv3d_2 (Conv3D)            (None, 5, 64, 64, 32)     27680     
    _________________________________________________________________
    conv3d_3 (Conv3D)            (None, 5, 64, 64, 32)     27680     
    _________________________________________________________________
    conv3d_4 (Conv3D)            (None, 3, 62, 62, 32)     27680     
    _________________________________________________________________
    conv3d_5 (Conv3D)            (None, 1, 60, 60, 32)     27680     
    _________________________________________________________________
    reshape_1 (Reshape)          (None, 60, 60, 32)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 60, 60, 4)         1156      
    =================================================================
    Total params: 112,772
    Trainable params: 112,772
    Non-trainable params: 0
    '''

    model.compile(loss = 'mean_squared_error', optimizer = adam(lr=0.001, decay=1e-6), metrics=[ssim_for2d, psnr_for2d])

    return model

