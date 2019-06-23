import numpy
import math
from PIL import Image
import cv2
'''
resample – An optional resampling filter.
This can be one of PIL.Image.NEAREST (use nearest neighbour),
PIL.Image.BILINEAR (linear interpolation),
PIL.Image.BICUBIC (cubic spline interpolation),
or PIL.Image.LANCZOS (a high-quality downsampling filter).
If omitted, or if the image has mode “1” or “P”, it is set PIL.Image.NEAREST.
'''

def getIndex(AMOUNT, TOTAL_AMOUNT):

  '''
  instead of taking the first i frames out from the video,
  pick i frames changed more sharply
  '''

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
  print('index for selecting frames')
  print(index)
  return index

    

def pickFrame(mode, i, path, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):

  filename = 'frame'+str(i)+'.tif'
    
  if mode == 'x':
    img = Image.open(path+'Frame_cmp/'+filename)
    data = img.resize((HEIGHT, WIDTH))
    data = data.resize((TARGET_HEIGHT, TARGET_WIDTH), Image.BICUBIC)

  elif mode == 'y':
    img = Image.open(path+'Frame_org/'+filename)
    data = data = img.resize((TARGET_HEIGHT, TARGET_WIDTH))
  
  data = numpy.array(data)

  img_yuv = cv2.cvtColor(data, cv2.COLOR_BGR2YUV)
  y, u, v = cv2.split(img_yuv)

  return y