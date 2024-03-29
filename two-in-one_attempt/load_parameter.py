#import scipy.io
from scipy.io import loadmat
import numpy as np
#import h5py

#filepath = '/home/user1/REUS/image-reconstruction/fine-tuned-model/9-1-5(91 images)/'
filepath = '/home/user1/REUS/image-reconstruction/fine-tuned-model/9-1-5(ImageNet)/'

#filename = ['x2.mat', 'x3.mat', 'x4.mat']  #for 9-1-5(91 images)
#filename = ['x3.mat']   # for 9-1-5(ImageNet)

def load_parameter(filename):
    para = loadmat(filepath + filename)
    ker_size = [9, 1, 5]
    ker_num = [1, 64, 32]

    f = []
    for i in range(1,4):
        biases = para['biases_conv{}'.format(i)]
        weights = para['weights_conv{}'.format(i)]
        weights = weights.reshape((ker_size[i-1], ker_size[i-1], ker_num[i-1], -1), order = 'F')
        biases = biases.reshape((-1,))
        f.append([weights, biases])

    f = np.asarray(f)
    
    return f
