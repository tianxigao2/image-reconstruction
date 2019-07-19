from load_data import *
import numpy
import pandas as pd
import matplotlib.pyplot as plt

# THE PATH IS USED FOR LOCAL DATASET FILES
# TO USE BRIDGES, USE THE getpath FUNTION STORED IN REMOTE COMPUTER

#LOCAL_PATH = '/home/user1/REUS/image-reconstruction/data/'
LOCAL_PATH = '/pylon5/ac5610p/janegao/image-reconstruction-2019/data/data/'

def save_model_path():
    return '/pylon5/ac5610p/janegao/image-reconstruction-2019/saved_model/'
# THE PATH IS USED FOR LOCAL DATASET FILES
# TO USE BRIDGES, USE THE getpath FUNTION STORED IN REMOTE COMPUTER

def Timofte():
    return '/pylon5/ac5610p/janegao/image-reconstruction-2019/data/data/Timofte_dataset/'

def get_Timofte(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):
    TMF = Timofte()
    return single_image_csv_set(AMOUNT, TOTAL_AMOUNT, TMF, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
    
#========================================================================================================
def climate():
    return (LOCAL_PATH + 'climate/', 2000)

def climate_2D(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):
    CLM, TOTAL_AMOUNT = climate()
    if AMOUNT > TOTAL_AMOUNT:
        print('Error: amount >  total_amount')
        return
    (x_set, y_set) = single_image_set(AMOUNT, TOTAL_AMOUNT, CLM, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
    (TARGET_HEIGHT, TARGET_WIDTH) = (318, 640)
    (HEIGHT, WIDTH) = (159, 320)
    return (x_set, y_set, TARGET_HEIGHT, TARGET_WIDTH, HEIGHT, WIDTH, AMOUNT, TOTAL_AMOUNT)   

