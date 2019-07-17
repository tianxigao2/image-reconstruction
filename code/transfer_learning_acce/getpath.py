from load_data import *
import numpy
import pandas as pd
import matplotlib.pyplot as plt

# THE PATH IS USED FOR LOCAL DATASET FILES
# TO USE BRIDGES, USE THE getpath FUNTION STORED IN REMOTE COMPUTER

LOCAL_PATH = '/home/user1/REUS/image-reconstruction/data/'

def prediction_path():
    return LOCAL_PATH + 'prediction_tmp/'

def save_model_path():
    return '/home/user1/REUS/image-reconstruction/saved_model/'

# for 3D SRNet model to take time-domain information -------------------------------------------------------------
def prediction_package(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, DEPTH):
    (AMOUNT, p_set) = packed_image_set_prediction(DEPTH, AMOUNT, TOTAL_AMOUNT, prediction_path(), HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
    (HEIGHT, WIDTH) = Image.open(prediction_path() + 'frame0.tif').size #set size only
    return (AMOUNT, p_set, HEIGHT, WIDTH)

# ==============================================================================================

def shepp_logan_phantom():
    return (LOCAL_PATH+'shepp logan phantom/', 2000)

# for 2D SRCNN model to do individual processing -----------------------------------------------------------------
def shepp_logan_phantom_2D(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):
    SLP, TOTAL_AMOUNT = shepp_logan_phantom()
    if AMOUNT < TOTAL_AMOUNT:
        print('Error: amount < total_amount')
        return
    (x_set, y_set) = single_image_set(AMOUNT, TOTAL_AMOUNT, SLP, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
    (TARGET_HEIGHT, TARGET_WIDTH) = (256, 256)
    (HEIGHT, WIDTH) = (256, 256)
    return (x_set, y_set, TARGET_HEIGHT, TARGET_WIDTH, HEIGHT, WIDTH)

# ================================================================================================================

def akiyo():
    return (LOCAL_PATH+'akiyo/', 300)

# for 2D SRCNN model to do individual processing -----------------------------------------------------------------
def akiyo_2D(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):
    AKY, TOTAL_AMOUNT = akiyo()
    if AMOUNT > TOTAL_AMOUNT:
        print('Error: amount > total_amount')
        return
    (x_set, y_set) = single_image_set(AMOUNT, TOTAL_AMOUNT, AKY, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
    (TARGET_HEIGHT, TARGET_WIDTH) = (352, 288)
    (HEIGHT, WIDTH) = (176, 144)
    return (x_set, y_set, TARGET_HEIGHT, TARGET_WIDTH, HEIGHT, WIDTH, AMOUNT, TOTAL_AMOUNT)

# ==================================================================================================================

def container():
    return (LOCAL_PATH+'container/', 300)

# for 2D SRCNN model to do individual processing -----------------------------------------------------------------
def container_2D(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):
    CTN, TOTAL_AMOUNT = container()
    if AMOUNT > TOTAL_AMOUNT:
        print('Error: amount > total_amount')
        return
    (x_set, y_set) = single_image_set(AMOUNT, TOTAL_AMOUNT, CTN, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)
    (TARGET_HEIGHT, TARGET_WIDTH) = (352, 288)
    (HEIGHT, WIDTH) = (176, 144)
    return (x_set, y_set, TARGET_HEIGHT, TARGET_WIDTH, HEIGHT, WIDTH, AMOUNT, TOTAL_AMOUNT)

# ===================================================================================================================

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

