'''
Created on 26 de mar de 2018

@author: marcelovca90
'''
import numpy as np
import random as rnd

class DataUtils:
    
    def __init__(self):
        pass
    
def add_bias(arr, bias = -1):
    for i in range(0, len(arr)):
        arr[i] = [bias] + arr[i]
    return arr

def shuffle(x, d):
    seq = rnd.sample(range(len(x)),len(x))
    i = 0
    for j in seq:
        tx = x[i]
        x[i] = x[j]
        x[j] = tx
        td = d[i]
        d[i] = d[j]
        d[j] = td
        i = i + 1
    return x,d

def split(arr, split_point = 0.5):
    first_half = arr[0:int(len(arr)*split_point)]
    second_half = arr[int(len(arr)*split_point):int(len(arr))]
    return first_half,second_half
