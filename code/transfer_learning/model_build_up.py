from acc import *
from load_parameter import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv3D
from keras import optimizers
from keras.optimizers import adam

def super_resolution_model(HEIGHT, WIDTH):
    
    model = Sequential()
    
    model.add(Conv2D(64, kernel_size=(9, 9), padding='same', activation='relu', use_bias = True,  input_shape=(HEIGHT,WIDTH,1)))
    model.add(Conv2D(32, (1, 1), padding='same', activation='relu', use_bias = True))
    model.add(Conv2D(1, (5, 5), padding='same', activation = 'relu', use_bias = True))
    
    #print(model.summary())

    model.compile(loss = 'mean_squared_error', optimizer = adam(lr=0.0005, decay=5e-5), metrics=[ssim_for2d, psnr_for2d])

    return model


def fine_tuned_SRCNN(FILE_INDEX, HEIGHT, WIDTH):
    model = Sequential()
    
    model.add(Conv2D(64, kernel_size=(9, 9), padding='same', activation='relu', use_bias = True,  input_shape=(HEIGHT,WIDTH,1)))
    model.add(Conv2D(32, (1, 1), padding='same', activation='relu', use_bias = True))
    model.add(Conv2D(1, (5, 5), padding='same', activation = 'relu', use_bias = True))

    for i in range(0,3):
        model.layers[i].set_weights(getParameter(FILE_INDEX)[i])
        print(model.layers[i].weights)
    
    #print(model.summary())
    
    model.compile(loss = 'mean_squared_error', optimizer = adam(lr=0.0005, decay=5e-5), metrics=[ssim_for2d, psnr_for2d])

    return model

def base_model_SRCNN(FILE_INDEX, HEIGHT, WIDTH):
    model = Sequential()
    
    model.add(Conv2D(64, kernel_size=(9, 9), padding='same', activation='relu', use_bias = True,  input_shape=(HEIGHT,WIDTH,1)))
    # model.add(Conv2D(32, (1, 1), padding='same', activation='relu', use_bias = True))
    
    for i in range(0,1):
        model.layers[i].set_weights(getParameter(FILE_INDEX)[i])
        print(model.layers[i].weights)
    
    print(model.summary())
    
    model.compile(loss = 'mean_squared_error', optimizer = adam(lr=0.001, decay=1e-6), metrics=[ssim_for2d, psnr_for2d])

    return model    

'''
def transfer_learning_SRCNN(FILE_INDEX, HEIGHT, WIDTH):
    base_model = base_model_SRCNN(FILE_INDEX, HEIGHT, WIDTH)
    base_model.trainable = False

    model = Sequential()
    model.add(base_model)
    
    model.add(Conv2D(32, (1, 1),padding = 'same', activation = 'relu', use_bias = True))
    model.layers[1].set_weights(getParameter(FILE_INDEX)[1])
    
    #model.add(Conv2D(32, (1, 1), padding='same', activation='relu', use_bias = True))
    
    model.add(Conv2D(1, (5, 5), padding='same', activation = 'relu', use_bias = True))
    model.layers[3].set_weights(getParameter(FILE_INDEX)[2])

    print(model.summary())

    model.compile(loss = 'mean_squared_error', optimizer = adam(lr=0.0029861308377023593, decay=5e-5), metrics=[ssim_for2d, psnr_for2d])

    return model
'''
def transfer_learning_SRCNN(filename, HEIGHT, WIDTH):

    par = load_parameter(filename)

    model = Sequential()
    
    model.add(Conv2D(64, kernel_size = (9,9), padding = 'same', activation = 'relu', use_bias = True,
    input_shape=(HEIGHT,WIDTH,1), trainable = False))
    model.add(Conv2D(32, (1, 1), padding='same', activation='relu', use_bias = True, trainable = True))
    model.add(Conv2D(1, (5, 5), padding='same', activation = 'relu', use_bias = True, trainable = True))

    for i in range(3):
        model.layers[i].set_weights(par[i])

    #print(model.summary())

    model.compile(loss = 'mean_squared_error', optimizer = adam(lr=0.0005, decay=1e-6), metrics=[ssim_for2d, psnr_for2d])

    return model

