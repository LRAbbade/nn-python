'''
Created on 5 de dez de 2017

@author: marcelovca90
'''
import numpy as np

def step(x):
    return -1 if x < 0 else +1

def eqm(w, x, d):
    eqm = 0
    for i in range(0,len(x)):
        v = np.dot(np.transpose(w), x[i])
        eqm = eqm + pow(d[i] - v, 2)
    eqm = eqm / len(x)
    return eqm
