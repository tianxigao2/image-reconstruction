import scipy.io
from scipy.io import loadmat
import numpy
import h5py

#filepath = '/home/user1/REUS/image-reconstruction-2019/papers and codes/SRCNN/SRCNN_test/model/9-1-5(91 images)/'
filepath = '/home/user1/REUS/image-reconstruction-2019/papers and codes/SRCNN/SRCNN_test/model/9-1-5(ImageNet)/'

#filename = ['x2.mat', 'x3.mat', 'x4.mat']  #for 9-1-5(91 images)
filename = ['x3.mat']   # for 9-1-5(ImageNet)

def getParameter(FILE_INDEX):
    # FILE_INDEX is used to choose the parameter storage loaded for our model

    f = []

    LAYERS = 3  #FOR SRCNN
    KERNAL_SIZE = [9, 1, 5] #FOR SRCNN
    IP_SIZE = [1, 64, 32]    #FOR SRCNN

    BIASES_CONV1 = loadmat(filepath + filename[FILE_INDEX])['biases_conv1']
    BIASES_CONV2 = loadmat(filepath + filename[FILE_INDEX])['biases_conv2']
    BIASES_CONV3 = loadmat(filepath + filename[FILE_INDEX])['biases_conv3']
    WEIGHTS_CONV1 = loadmat(filepath + filename[FILE_INDEX])['weights_conv1']
    WEIGHTS_CONV2 = loadmat(filepath + filename[FILE_INDEX])['weights_conv2']
    WEIGHTS_CONV3 = loadmat(filepath + filename[FILE_INDEX])['weights_conv3']

    par = [BIASES_CONV1, BIASES_CONV2, BIASES_CONV3, WEIGHTS_CONV1, WEIGHTS_CONV2, WEIGHTS_CONV3]

    for j in range(LAYERS):
        WEIGHT_IN_LAYER_J = par[j + LAYERS]
        BIAS_IN_LAYER_J = par[j]

        tmp_weight = numpy.asarray(WEIGHT_IN_LAYER_J).reshape((KERNAL_SIZE[j], KERNAL_SIZE[j], IP_SIZE[j], -1))#.transpose(1, 0, 2, 3)
        tmp_bias = numpy.asarray(BIAS_IN_LAYER_J).reshape((-1,))
        
        f.append([tmp_weight, tmp_bias]) # append [weight_in_layer_j, bias_in_layer_j]

    f = numpy.asarray(f)

    return f
