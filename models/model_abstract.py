import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class base_model:
    def __init__(self):
        pass
        
    def fit(self, datax, datay = None):
        pass
    
    def predict(self, x_test):
        pass
    
    def score(self, func_test, x_test):
        return func_test(x_test)