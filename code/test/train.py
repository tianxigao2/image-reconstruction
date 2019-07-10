from acc import *
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# The patience parameter is the amount of epochs to check for improvement
#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
'''

    monitor: quantity to be monitored.
    min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement.
    patience: number of epochs with no improvement after which training will be stopped.
    verbose: verbosity mode.
    mode: one of {auto, min, max}. In min mode, training will stop when the quantity monitored has stopped decreasing; in max mode it will stop when the quantity monitored has stopped increasing; in auto mode, the direction is automatically inferred from the name of the monitored quantity.
    baseline: Baseline value for the monitored quantity to reach. Training will stop if the model doesn't show improvement over the baseline.
    restore_best_weights: whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used.

'''
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.plot(hist['epoch'], hist['psnr_for2d'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_psnr_for2d'], label = 'Val Error')
    plt.ylim([0,10])
    plt.legend()
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.plot(hist['epoch'], hist['ssim_for2d'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_ssim_for2d'], label = 'Val Error')
    plt.ylim([0,2.5])
    plt.legend()

    plt.show()
