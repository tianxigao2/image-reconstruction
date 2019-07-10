# -*- coding: utf-8 -*-
import numpy
import math
from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt
'''
resample – An optional resampling filter.
This can be one of PIL.Image.NEAREST (use nearest neighbour),
PIL.Image.BILINEAR (linear interpolation),
PIL.Image.BICUBIC (cubic spline interpolation),
or PIL.Image.LANCZOS (a high-quality downsampling filter).
If omitted, or if the image has mode “1” or “P”, it is set PIL.Image.NEAREST.
'''

# getIndex is used to take the i average-splitted frames out -----------------------------------------------------------
# Used in single image data set
# For example, Timofte
# Locally running shepp-logan phantom may also requires index pick-up, so not deleted
def getIndex(AMOUNT, TOTAL_AMOUNT):

  prop = math.floor(TOTAL_AMOUNT/AMOUNT) #because of math.floor, (i * prop) should be equal or smaller to TOTAL_AMOUNT
  
  all = range(TOTAL_AMOUNT)
  take = all[::prop]

  count = AMOUNT - len(take)
  appendIn = 2
  index = take
  while(count != 0):
    if count < 0:   #more pictures picked than asked, cut the latest few elements
      index = take[0:AMOUNT]
    if count > 0:   #less pictures picked than asked, append with the first few skipped elements
      take.append(appendIn)
      appendIn += 1
      index = take
    count = AMOUNT - len(index)

  # after while-loop return, should get exactly AMOUNT of picture index in the list "index"
  #print('index for selecting frames', index)
  return index


# pickFrame and pickCSV are used for single image datatype loading and processing ----------------------------------------------------------------------

# loadCSV works for data stored as csv file ==================================================================
# The function will read data as a numpy ndarray
def loadCSV(mode, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):

  if mode == 'x':
    filename = 'training_data_y.csv'
  elif mode == 'y':
    filename = 'ground_truth_y.csv'
  
  tmp = pd.read_csv(path+filename, header=None)
  data = tmp.values.reshape((-1, TARGET_HEIGHT, TARGET_WIDTH, 1), order = 'F')
  print('loading the whole csv file', data.shape)
  return data

# pickCSV will pick certain frames out from the whole dataset
def pickCSV(AMOUNT, TOTAL_AMOUNT, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):
  x_data = loadCSV('x', path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
  y_data = loadCSV('y', path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
  x_set = []
  y_set = []
  for i in getIndex(AMOUNT, TOTAL_AMOUNT):
    x_tmp = x_data[i, :, :, 0]
    x_set.append([x_tmp])
    y_tmp = y_data[i, :, :, 0]
    y_set.append([y_tmp])
  return(numpy.array(x_set), numpy.array(y_set))


# single_image_set is for the final packing of single-image processing case --------------------------------------------------
# The function will reshape the final output into 4d tensor (the last dim stores color channel)
def single_image_csv_set(AMOUNT, TOTAL_AMOUNT, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):
  (x_set, y_set) = pickCSV(AMOUNT, TOTAL_AMOUNT, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
  print('shape before reshape', x_set.shape)
  x_set = numpy.reshape(x_set, (AMOUNT, TARGET_HEIGHT, TARGET_WIDTH, 1))
  y_set = numpy.reshape(y_set, (AMOUNT, TARGET_HEIGHT, TARGET_WIDTH, 1))
  return(x_set, y_set)