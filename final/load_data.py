# -*- coding: utf-8 -*-
import numpy
import h5py
import scipy
from scipy import misc
import cv2
import math
from PIL import Image

def load_h5_data(filename, datasetname, AMOUNT):
    f = h5py.File(filename, 'r') 
    data = f[datasetname][-AMOUNT:,:,:,:]
    [AMOUNT, HEIGHT, WIDTH, tmp] = list(data.shape)
    return (data, AMOUNT, HEIGHT, WIDTH)

def load_h5_data_testset(filename, datasetname, START_INDEX, END_INDEX):
    f = h5py.File(filename, 'r') 
    data = f[datasetname][START_INDEX:END_INDEX,:,:,:]
    [AMOUNT, HEIGHT, WIDTH, tmp] = list(data.shape)
    return (data, AMOUNT, HEIGHT, WIDTH)

def enlarge(x_set, TARGET_HEIGHT, TARGET_WIDTH):
    [AMOUNT, HEIGHT, WIDTH, tmp2] = x_set.shape
    x_enlarge = []
    for i in range(AMOUNT):
        frame = x_set[i, :, :].reshape((HEIGHT, WIDTH))
#        frame_enlarge = scipy.misc.imresize(numpy.asarray(frame), (TARGET_HEIGHT, TARGET_WIDTH), interp='bicubic', mode='L')
#        frame_enlarge = numpy.asarray(frame).resize((TARGET_HEIGHT, TARGET_WIDTH),Image.BICUBIC)
#        frame_enlarge = cv2.resize(numpy.asarray(frame), None, fx = 2, fy = 2)
        frame_enlarge = cv2.resize(numpy.asarray(frame), dsize=(TARGET_HEIGHT, TARGET_WIDTH), interpolation=cv2.INTER_NEAREST)	#INTER_CUBIC
        x_enlarge.append(frame_enlarge)
    x_enlarge = numpy.reshape(x_enlarge,(AMOUNT, TARGET_HEIGHT, TARGET_WIDTH, 1))
    return x_enlarge

def repacking(x):
    #----------------------------get shape from input numpy array---------------------------------
    (AMOUNT, TARGET_HEIGHT, TARGET_WIDTH, tmp) = x.shape
    DEPTH = 5
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
                frame = x[j, :, :]
                frame = numpy.reshape(frame,(1, TARGET_HEIGHT, TARGET_WIDTH))
                
                if j == i - HALF_RANGE:
                    package = frame
                else:   
                    package = numpy.append(package, frame, axis = 0)

            if FIRST == True:
                package_set = package
                FIRST = False
            else:
                package_set = numpy.append(package_set, package, 0)

    package_set = numpy.reshape(package_set, (AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH, 1))
    return package_set

