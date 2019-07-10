import h5py
import numpy
from PIL import Image


filePath = '/home/user1/REUS/image-reconstruction-2019/Video_Stream/H_Field_Nl5.mat'
orgPicPath = '/home/user1/REUS/image-reconstruction-2019/data/climate/Frame_org/'
cmpPicPath = '/home/user1/REUS/image-reconstruction-2019/data/climate/Frame_cmp/'

f = h5py.File(filePath,'r')    #<HDF5 file "H_Field_Nl5.mat" (mode r)>
# list(f.keys()):  ['H_Field']
# f['H_Field']:   <HDF5 dataset "H_Field": shape (2000, 640, 318), type "<f8">
data = f['H_Field'][:]      #<class 'numpy.ndarray'>, (2000, 640, 318)

TOTAL_AMOUNT, HEIGHT, WIDTH = 2000, 640, 318
DOWNGRADE_HEIGHT = HEIGHT // 2
DOWNGRADE_WIDTH = WIDTH // 2

for i in range(TOTAL_AMOUNT):
    img = data[i,:,:]
    img = Image.fromarray(img)
    
    img.save(orgPicPath + 'frame' + str(i) + '.tif')
    
    # downsize the image with an ANTIALIAS filter (gives the highest quality)
    img = img.resize((DOWNGRADE_HEIGHT,DOWNGRADE_WIDTH))
    
    img.save(cmpPicPath + 'frame' + str(i) + '.tif')    #, optimize = True, quality = 50
    print('save ' + 'frame' + str(i) + '.tif')

'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread(cmpPicPath + 'frame' + '0' + '.tif')
imgplot = plt.imshow(img)
plt.show()
'''
