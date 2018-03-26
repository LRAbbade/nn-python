'''
Created on 9 de mar de 2018

@author: marcelocysneiros
'''
import matplotlib.pyplot as plt
import numpy as np

class PlotUtils:
    
    def __init__(self):
        pass
    
# https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes
def plot(x, _xlabel, y, _ylabel, e=None):
    
    # data
    ax = plt.gca()
    if e is not None:
        plt.plot([1, len(x)], [e, e], color='red', linestyle='dashed', linewidth=1.0)
    ax.plot(x, y, color='blue', linewidth=1.5)
    
    # limits
    ax.set_xlim([1, len(x)])
    ax.set_ylim([0, max(y) * 1.1])
    
    # text
    ax.set_xlabel(_xlabel)
    ax.set_ylabel(_ylabel)
    ax.set_title('{} vs {}'.format(_xlabel, _ylabel))
    
    # display
    ax.grid()
    plt.show()
