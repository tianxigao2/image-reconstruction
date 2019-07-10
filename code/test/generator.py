import math
import numpy as np


class hyperparameter(dict):
    def __init__(self):
        self = {}
        
    def add(self, hypa_name: str, hypa_type = 'float', hypa_range = [0,1] ):
        self[hypa_name] = [hypa_type, hypa_range]
        
    def single_generator(self) -> list:
        choice = []
        for item in self:
            if self[item][0] == 'float':
                upper = self[item][1][1]
                lower = self[item][1][0]
                choice.append((upper-lower)*np.random.random()+lower)
            if self[item][0] == 'int':
                upper = self[item][1][1]
                lower = self[item][1][0]
                choice.append(np.random.randint(lower, upper))
            if self[item][0] == 'list':
                choice.append(np.random.choice(self[item][1]))
        return choice
    
    def multi_generator(self, size_0 :int):
        chosen = {}
        for item in self:
            if self[item][0] == 'float':
                upper = self[item][1][1]
                lower = self[item][1][0]
                chosen[item] = (upper-lower)*np.random.random(size_0,)+lower
            if self[item][0] == 'int':
                upper = self[item][1][1]
                lower = self[item][1][0]
                chosen[item] = np.random.randint(lower, upper, size_0)
            if self[item][0] == 'list':
                chosen[item] = np.random.choice(self[item][1],size_0)
        return chosen
    
