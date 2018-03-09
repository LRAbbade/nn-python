'''
Created on 5 de dez de 2017

@author: marcelovca90
'''

class SampleData:
    
    def __init__(self):
        pass
    
class AND:
    input = [ [ -1, -1 ], [ -1, +1 ], [ +1, -1 ], [ +1, +1 ] ];
    output = [ -1, -1, -1, +1 ];

class OR:
    input = [ [ -1, -1 ], [ -1, +1 ], [ +1, -1 ], [ +1, +1 ] ];
    output = [ -1, +1, +1, +1 ];

class XOR:
    input = [ [ -1, -1 ], [ -1, +1 ], [ +1, -1 ], [ +1, +1 ] ];
    output = [ -1, +1, +1, -1 ];
