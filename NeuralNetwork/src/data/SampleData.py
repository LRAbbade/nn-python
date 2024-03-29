'''
Created on 5 de dez de 2017

@author: marcelovca90
'''

import numpy as np
import os, sys
from numpy.random import sample
from numpy import genfromtxt

class SampleData:
    
    @staticmethod
    def read(folder, filename, flatten = False):
        filename_abs = os.path.join(os.path.dirname(__file__), folder, filename)
        return genfromtxt(filename_abs, delimiter=',', dtype=float)

# https://en.wikipedia.org/wiki/AND_gate
class LOGIC_GATE_AND:
    input = SampleData.read('logic-gate-and', 'input.txt')
    output = SampleData.read('logic-gate-and', 'output.txt', True)

# https://en.wikipedia.org/wiki/OR_gate
class LOGIC_GATE_OR:
    input = SampleData.read('logic-gate-or', 'input.txt')
    output = SampleData.read('logic-gate-or', 'output.txt', True)

# https://en.wikipedia.org/wiki/XOR_gate
class LOGIC_GATE_XOR:
    input = SampleData.read('logic-gate-xor', 'input.txt')
    output = SampleData.read('logic-gate-xor', 'output.txt', True)

# https://archive.ics.uci.edu/ml/datasets/ionosphere
class IONOSPHERE:
    input = SampleData.read('ionosphere', 'input.txt')
    output = SampleData.read('ionosphere', 'output.txt', True)

# https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
class TIC_TAC_TOE_ENDGAME:
    input = SampleData.read('tic-tac-toe-endgame', 'input.txt')
    output = SampleData.read('tic-tac-toe-endgame', 'output.txt', True)
    
# http://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
class BLOOD_TRANSFUSION:
    input = SampleData.read('blood-transfusion', 'input.txt')
    output = SampleData.read('blood-transfusion', 'output.txt', True)

# 
class DIABETES:
    input = SampleData.read('diabetes', 'input.txt')
    output = SampleData.read('diabetes', 'output.txt', True)
