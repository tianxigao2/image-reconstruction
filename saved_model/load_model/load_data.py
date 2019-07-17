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

# pickFrame works for data store as images  (shepp-logan-phantom.tif)
# The function will convert the image into YUV model, and take out the Y channel only
# The function will return a <non-type>; to use the output, wrap it with "numpy.array()"
def pickFrame(mode, i, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, layer_3d):

  filename = 'frame'+str(i)+'.tif'
    
  #img = Image.open(path+'Frame_org/'+filename)
  #data = img.resize((TARGET_HEIGHT, TARGET_WIDTH))
  if mode == 'x':
    img = Image.open(path + 'Frame_cmp/' + filename)
    
    if (HEIGHT, WIDTH) == (TARGET_HEIGHT, TARGET_WIDTH):  # for sehpp-logan phantom set, firstly smaller the image size
      HEIGHT = HEIGHT//2
      WIDTH = WIDTH//2
      img = img.resize((HEIGHT, WIDTH))
    
    data = img.resize((TARGET_HEIGHT, TARGET_WIDTH),Image.BICUBIC)
    data_tmp = numpy.asarray(data)
    data = data_tmp

  else:
    if mode == 'y':
      img = Image.open(path + 'Frame_org/' + filename)
    
    if mode == 'p': #prediction from interim result
      img = Image.open(path + filename)
    
    data_tmp = numpy.asarray(img)
    data = data_tmp

  if layer_3d == True and mode == 'y':
    #need to make y_set picture size smaller
    data = data[2:TARGET_HEIGHT-2, 2:TARGET_WIDTH-2]
  
  #print('shape',data.shape)
  #print('target_height: ', TARGET_HEIGHT, ', height: ', HEIGHT)
  #print('data.ndim: ', data.ndim)

  if data.ndim == 2:  #prediction get only one channel y
    return data 
  
  img_yuv = cv2.cvtColor(data, cv2.COLOR_BGR2YUV) #--just backup for channel != 1 cases--
  y, u, v = cv2.split(img_yuv)

  return y

# pack is the first step of multi-image processing -------------------------------------------------------------------
# The function will take "DEPTH" pictures out and pack as one small package
# The function will return a ndarray with a shape of (DEPTH, TARGET_HEIGHT, TARGET_WIDTH, CHANNEL)
def pack(DEPTH, mode, i, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, TOTAL_AMOUNT):
  
  HALF_RANGE = math.floor(DEPTH/2)
  data = []
  if DEPTH%2 == 0:
    RANGE = range(i - HALF_RANGE, i + HALF_RANGE)
  else:
    RANGE = range(i - HALF_RANGE, i + HALF_RANGE + 1) 

  for j in RANGE:
    tmp = pickFrame(mode, j, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, True).tolist()  
    #<class 'numpy.ndarray'> (64, 64)
    data.append(tmp)
    
  data = numpy.array(data)
  data = numpy.reshape(data, (DEPTH, TARGET_HEIGHT, TARGET_WIDTH, 1))
  return data  #return a single package

#packed_image_set is to pack all the packages into a dataset --------------------------------------------------------
# only as a backup, not used in 2d+3d model program
def packed_image_set(DEPTH, AMOUNT, TOTAL_AMOUNT, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):

  x_set = []

  for i in getIndex(AMOUNT, TOTAL_AMOUNT):
    
    HALF_RANGE = math.floor(DEPTH/2)

    if (i - HALF_RANGE) < 0 or (i + HALF_RANGE) >= TOTAL_AMOUNT:
      AMOUNT = AMOUNT - 1
      
    else:
      x_set.append(pack(DEPTH, 'x', i, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, TOTAL_AMOUNT).tolist())
      
  x_set = numpy.reshape(numpy.array(x_set), (AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH, 1))

  y_set = numpy.array([pickFrame('y', i, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, True) for i in getIndex(AMOUNT, TOTAL_AMOUNT)])
  y_set = numpy.reshape(y_set, (AMOUNT, TARGET_HEIGHT - 2 * 2, TARGET_WIDTH - 2 * 2, 1))
    
  return (AMOUNT, x_set, y_set)

#function for picking up interim resulted pictures---------------------------------------------------------
def packed_image_set_prediction(DEPTH, AMOUNT, TOTAL_AMOUNT, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):

  p_set = []

  for i in getIndex(AMOUNT, TOTAL_AMOUNT):
    
    HALF_RANGE = math.floor(DEPTH/2)

    if (i - HALF_RANGE) < 0 or (i + HALF_RANGE) >= TOTAL_AMOUNT:
      AMOUNT = AMOUNT - 1
      
    else:
      p_set.append(pack(DEPTH, 'p', i, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, TOTAL_AMOUNT).tolist())
      print('packed '+ str(i) + ' packages already')    
      
  p_set = numpy.reshape(numpy.array(p_set), (AMOUNT, DEPTH, TARGET_HEIGHT, TARGET_WIDTH, 1))
  print('finish picking up prediction set!')

  #y_set = numpy.array([pickFrame('y', i, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, True) for i in getIndex(AMOUNT, TOTAL_AMOUNT)])
  #y_set = numpy.reshape(y_set, (AMOUNT, TARGET_HEIGHT - 2 * 2, TARGET_WIDTH - 2 * 2, 1))
    
  return (AMOUNT, p_set)  #, y_set)

# single_image_set is for the final packing of single-image processing case --------------------------------------------------
# The function will reshape the final output into 4d tensor (the last dim stores color channel)
def single_image_set(AMOUNT, TOTAL_AMOUNT, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):
  x_set = numpy.array([pickFrame('x', i, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, False) for i in getIndex(AMOUNT, TOTAL_AMOUNT)])
  x_set = numpy.reshape(x_set, (AMOUNT, TARGET_HEIGHT, TARGET_WIDTH, 1))
  y_set = numpy.array([pickFrame('y', i, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, False) for i in getIndex(AMOUNT, TOTAL_AMOUNT)])
  y_set = numpy.reshape(y_set, (AMOUNT, TARGET_HEIGHT, TARGET_WIDTH, 1))
  return(x_set, y_set)
 