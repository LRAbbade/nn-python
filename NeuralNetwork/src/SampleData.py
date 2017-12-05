'''
Created on 5 de dez de 2017

@author: marcelovca90
'''
class AND:
    input = [ [ -1, -1 ], [ -1, +1 ], [ +1, -1 ], [ +1, +1 ] ];
    output = [ -1, -1, -1, +1 ];

class OR:
    input = [ [ -1, -1 ], [ -1, +1 ], [ +1, -1 ], [ +1, +1 ] ];
    output = [ -1, +1, +1, +1 ];

class XOR:
    input = [ [ -1, -1 ], [ -1, +1 ], [ +1, -1 ], [ +1, +1 ] ];
    output = [ -1, +1, +1, -1 ];
    
def add_bias(v, bias):
    for i in range(0, len(v)):
        v[i] = [bias] + v[i]
    return v
