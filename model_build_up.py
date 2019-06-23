from acc import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.optimizers import adam
from PSNR1 import *

def sequential_model(HEIGHT, WIDTH):
    
    model = Sequential()
    
    model.add(Conv2D(64, kernel_size=(9, 9), padding='same', activation='relu', use_bias = True,  input_shape=(HEIGHT,WIDTH,1)))
    model.add(Conv2D(32, (1, 1), padding='same', activation='relu', use_bias = True))
    model.add(Conv2D(1, (5, 5), padding='same', activation = 'relu', use_bias = True))

    #model.add(Reshape(AMOUNT, TARGET_HEIGHT, TARGET_WIDTH,-1))
    #model.add(Flatten()) #(257, 512, 512)
    #model.add(Dropout(0.25))
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dense(1))
    
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

    #example_batch = train[:10]
    #example_result = model.predict(example_batch)
    #print(example_result)

    return model

