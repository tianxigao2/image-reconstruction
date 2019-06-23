import cv2

def extractor(VideoName, ImagePath):
    vidcap = cv2.VideoCapture(VideoName)
    success,image = vidcap.read()
    count = 0
    print('in')
    while success:
        imageName = ImagePath + "frame%d.tif"% count
        cv2.imwrite(imageName , image)     # save frame as TIF file to avoid another compression      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

Video_compress = "/home/user1/REUS/image-reconstruction-2019/data/compressedSample_50.avi"
Frame_compress = '/home/user1/REUS/image-reconstruction-2019/data/Frame_cmp/'

Video_org = "/home/user1/REUS/image-reconstruction-2019/data/uncompressedSample.avi"
Frame_org = "/home/user1/REUS/image-reconstruction-2019/data/Frame_org/"

extractor(Video_compress, Frame_compress)
extractor(Video_org, Frame_org)

