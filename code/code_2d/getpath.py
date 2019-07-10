from load_data import *
import numpy
import pandas as pd
import matplotlib.pyplot as plt

# THE PATH IS USED FOR LOCAL DATASET FILES
# TO USE BRIDGES, USE THE getpath FUNTION STORED IN REMOTE COMPUTER

def Timofte():
    return '/home/user1/REUS/image-reconstruction-2019/data/Timofte_dataset/'

def get_Timofte(AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH):
    TMF = Timofte()
    return single_image_csv_set(AMOUNT, TOTAL_AMOUNT, TMF, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH)

