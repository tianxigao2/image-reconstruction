from acc import *
from load_parameter import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras import optimizers
from keras.optimizers import adam



def transfer_learning_SRCNN_51311(filename, HEIGHT, WIDTH):

    (weights, biases, prelu) = load_parameter(filename)
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size = (5,5), padding = 'same', use_bias = True, input_shape=(HEIGHT,WIDTH,1), activation = 'relu', trainable = False))
    #keras.layers.LeakyReLU(alpha = prelu[0][0])
    model.add(Conv2D(5, (1,1), padding = 'same', use_bias = True, activation = 'relu', trainable = False))
    #keras.layers.LeakyReLU(alpha = prelu[1][0])
    model.add(Conv2D(5, (3,3), padding = 'same', use_bias = True, activation = 'relu', trainable = False))
    #keras.layers.LeakyReLU(alpha = prelu[2][0])
    model.add(Conv2D(32, (1,1), padding = 'same', use_bias = True, activation = 'relu', trainable = True))
    #keras.layers.LeakyReLU(alpha = prelu[3][0])
    model.add(Conv2DTranspose(1, (2,2), strides = (2,2), activation = 'relu'))
    #keras.layers.PReLU(alpha_initializer='zeros')

    for i in range(0,4):
        model.layers[i].set_weights([weights[i], biases[i]])

    #print(model.summary())

    model.compile(loss = 'mean_squared_error', optimizer = adam(lr=0.005, decay=1e-8), metrics=[ssim_for2d, psnr_for2d])

    return model
