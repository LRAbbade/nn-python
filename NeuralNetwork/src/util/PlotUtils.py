'''
Created on 9 de mar de 2018

@author: marcelocysneiros
'''
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

class PlotUtils:
    
    def __init__(self):
        pass
    
# https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes
def plot(x, _xlabel, y, _ylabel):
    
    # handle convergence in first training epoch
    if len(x) == 1 and len(y) == 1:
        print('Network converged in one training epoch; no plot to show.')
    else:
        # data
        ax = plt.gca()
        ax.plot(x, y, color='blue', linewidth=1.5)
        
        # text
        ax.set_xlabel(_xlabel)
        ax.set_ylabel(_ylabel)
        ax.set_title('{} vs {}'.format(_xlabel, _ylabel))
        
        # display
        ax.grid()
        date_string = dt.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
        plt.savefig('{}.png'.format(date_string))
        plt.autoscale(enable=True, axis='y')
        plt.show()
