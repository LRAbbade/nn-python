'''
Created on 26 de mar de 2018

@author: marcelovca90
'''
import numpy as np

class DataUtils:

    def __init__(self):
        pass

def add_bias(arr, bias = -1):
    ans = np.empty([len(arr), len(arr[0])+1])
    for i in range(0, len(arr)):
        ans[i] = np.insert(arr[i], 0, bias)
    return ans

def shuffle(x, d):
    k = len(x)
    seq = np.random.choice(range(k), k, False)
    i = 0
    for j in seq:
        x[[i,j]] = x[[j,i]]
        d[[i,j]] = d[[j,i]]
        i = i + 1
    return x,d

def splitTrainTest(arr, training_percentage = 0.5):
    training = arr[0:int(len(arr)*training_percentage)]
    test = arr[int(len(arr)*training_percentage):int(len(arr))]
    return training,test

def splitTrainValidateTest(arr, training_percent = 0.4, validation_percent = 0.2):
    training = arr[0:int(len(arr)*training_percent)]
    validation = arr[int(len(arr)*training_percent):int(len(arr)*(training_percent+validation_percent))]
    test = arr[int(len(arr)*(training_percent+validation_percent)):int(len(arr))]
    return training,validation,test

def random_seed():
    return np.random.randint(65536);
