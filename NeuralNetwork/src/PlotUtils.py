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
def plot(x, _xlabel, y, _ylabel):
    
    # data
    ax = plt.gca()
    ax.plot(x, y, color='blue', marker='o', mec='red', mfc='red', linestyle='dashed', linewidth=1.5, markersize=4.5)
    
    # limits
    ax.set_xlim([1, len(x)])
    ax.set_ylim([0, max(y)])
    
    # ticks        
    ax.set_xticks(x)
    ax.set_yticks(np.linspace(0, max(y), len(x)))
    
    # text
    ax.set_xlabel(_xlabel)
    ax.set_ylabel(_ylabel)
    ax.set_title('{} vs {}'.format(_xlabel, _ylabel))
    
    # display
    ax.grid()
    plt.show()
