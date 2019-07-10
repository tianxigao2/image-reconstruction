from typing import List
from tensorflow import keras
import numpy as np

# design metric for early stopping
def my_metric(the_data_metric: np.ndarray, the_data_perform: List[float], new_history: np.ndarray, 
              num_nearest_nbrs = 1, modified_rate = 1, keep_rate = 0.5 ) -> bool:
  # find the k nearest trials and use average performance of these neighbors to predict the ongoing one
  # if ranking of the modified performance below the keep_rate, return True; otherwise, return False
    length = len(new_history)
    predict = 0
    distance_line = np.linalg.norm(the_data_metric[...,:length] - new_history, ord = 2, axis = 1)
    for item in list(np.argpartition(distance_line, num_nearest_nbrs))[:num_nearest_nbrs]:
        predict += the_data_perform[item]
    predict = predict / num_nearest_nbrs
    pre_order = 1
    for item in the_data_perform:
        pre_order += (item * modified_rate > predict)
    return pre_order / len(the_data_perform) < (1 - keep_rate)

class NewMetricMeasure(keras.callbacks.Callback):
    def __init__(self, x_test, y_test, trials:int, split_rate = 0.2, the_data_metric = np.array([]), the_data_perform = [], keep_rate = 0.5, 
                metric_name = 'loss', boundary = [0, 0], checkpts = [], cum_pts = np.linspace(0.01, 1, num = 1000, endpoint = False)):
      # boundary is a list including the lower and upper percentages for my_metric checking
      # boundary can be replaced by a set of checkpoints represented as precetages
      self.history_metric = the_data_metric
      self.history_perform = the_data_perform
      self.raw_boundary = boundary
      self.raw_checkpts = checkpts
      self.raw_cum_pts = cum_pts
      self.keep_rate = keep_rate
      self.x_test = x_test
      self.y_test = y_test
      self.metric_name = metric_name
      self.split_rate = split_rate
      self.trials = trials
      self.best_trial = 0
      self.running_trial = -1
      self.best_performance = 100
      self.best_model = None
      
    def on_train_begin(self, logs={}):
      self.running_trial += 1
      self.metric_data = np.array([])
      self.check = (len(self.history_perform) > round(self.trials * self.split_rate))    
      self.steps = 0
      if not 'steps_per_epoch' in self.params:
        self.params['steps_per_epoch'] = self.params['samples'] // self.params['batch_size'] + 1
      self.total_number = self.params['steps_per_epoch'] * self.params['epochs']
      self.cum_pts = np.round_(np.array(self.raw_cum_pts) * self.total_number)
      self.boundary = np.array(self.raw_boundary) * self.total_number
      self.checkpts = np.round_(np.array(self.raw_checkpts) * self.total_number)

    def on_batch_end(self, batch, logs={}):
        self.steps += 1
        if self.steps in list(self.cum_pts):
            self.metric_data = np.append(self.metric_data, logs.get(self.metric_name))
        if self.check and ((self.boundary[0] < batch < self.boundary[1]) or (self.steps in self.checkpts)):
            self.model.stop_training = my_metric(self.history_metric, self.history_perform, self.metric_data, keep_rate  = self.keep_rate)

    def on_epoch_end(self, epoch, logs={}):
        if logs.get(self.metric_name) > 0.25 and epoch >= 3:
            self.model.stop_training = True

    def on_train_end(self, logs={}):
        if not self.model.stop_training:
            try:
              self.history_metric = np.row_stack((self.history_metric, self.metric_data))
            except Exception:
              self.history_metric = self.metric_data
            mean_squared_error, ssim_trained, psnr_trained = self.model.evaluate(self.x_test, self.y_test, verbose=2)
            self.history_perform.append(mean_squared_error)
            if self.best_performance > mean_squared_error:
              self.best_performance = mean_squared_error
              self.best_trial = self.running_trial
              self.best_model = self.model
