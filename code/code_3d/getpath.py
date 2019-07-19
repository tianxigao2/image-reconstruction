from load_data import *
import numpy
import pandas as pd
import matplotlib.pyplot as plt

# THE PATH IS USED FOR LOCAL DATASET FILES
# TO USE BRIDGES, USE THE getpath FUNTION STORED IN REMOTE COMPUTER

#LOCAL_PATH = '/home/user1/REUS/image-reconstruction-2019/data/'
LOCAL_PATH = '/pylon5/ac5610p/janegao/image-reconstruction-2019/data/data/'

# ==============================================================================================

def shepp_logan_phantom():
    return (LOCAL_PATH+'shepp logan phantom/', 2000)

def get_shepp_logan_phantom_2D(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):
    SLP, TOTAL_AMOUNT = shepp_logan_phantom()
    if AMOUNT < TOTAL_AMOUNT:
        print('Error: amount < total_amount')
        return
    return single_image_set(AMOUNT, TOTAL_AMOUNT, SLP, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)

def shepp_logan_phantom_package(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, DEPTH):
    SLP, TOTAL_AMOUNT = shepp_logan_phantom()
    return packed_image_set(DEPTH, AMOUNT, TOTAL_AMOUNT, SLP, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
    
# ================================================================================================================

def akiyo():
    return (LOCAL_PATH+'akiyo/', 300)

def get_akiyo_2D(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):
    AKY, TOTAL_AMOUNT = akiyo()
    if AMOUNT < TOTAL_AMOUNT:
        print('Error: amount < total_amount')
        return
    return single_image_set(AMOUNT, TOTAL_AMOUNT, AKY, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)

def akiyo_package(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, DEPTH):
    AKY, TOTAL_AMOUNT = akiyo()
    return packed_image_set(DEPTH, AMOUNT, TOTAL_AMOUNT, AKY, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
    
# ==================================================================================================================

def container():
    return (LOCAL_PATH+'container/', 300)

def get_container_2D(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):
    CTN, TOTAL_AMOUNT = container()
    if AMOUNT < TOTAL_AMOUNT:
        print('Error: amount < total_amount')
        return
    return single_image_set(AMOUNT, TOTAL_AMOUNT, CTN, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)

def container_package(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, DEPTH):
    CTN, TOTAL_AMOUNT = container()
    return packed_image_set(DEPTH, AMOUNT, TOTAL_AMOUNT, CTN, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)

    
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

