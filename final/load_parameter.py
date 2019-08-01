from scipy.io import loadmat
import numpy as np

def load_parameter_FSRCNN(filename):
    par = loadmat(filename)
    ker_size = [5, 1, 3, 1, 9]
    ker_num = [1, 32, 5, 5, 1]

    par_name = ['weights_conv', 'biases_conv', 'prelu_conv']
    weights_conv = par[par_name[0]]
    biases_conv = par[par_name[1]]
    prelu_conv = par[par_name[2]]
    weights = []
    biases = []
    prelu = []
    
    for i in range(0,5):
        tmp = weights_conv[i][0].reshape((ker_size[i], ker_size[i], ker_num[i], -1), order = 'F')
        weights.append(tmp)

    for i in range(0,5):
        tmp = biases_conv[i][0].reshape((-1, ))
        biases.append(tmp)

    for i in range(0,5):
        tmp = prelu_conv[i][0].reshape((-1, ))
        prelu.append(tmp)
    
    return (weights, biases, prelu)

def load_parameter(filename):
    para = loadmat(filename)
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