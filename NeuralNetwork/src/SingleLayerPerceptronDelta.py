'''
Created on 5 de dez de 2017

@author: marcelovca90
'''
import numpy as np
import SampleData
import MathUtils

class SingleLayerPerceptronDelta:

    def __init__(self):
        print('SingleLayerPerceptronDelta')

    def train(self, x, d, n, e):
        k = len(x[0])
        w = np.random.rand(k)
        epoch = 0
        while True:
            eqm_prev = MathUtils.eqm(w, x, d)
            for i in range(0, k):
                v = np.dot(np.transpose(w), x[i])
                w = np.add(w, np.multiply(x[i], n * (d[i] - v)))
            epoch = epoch + 1
            eqm_curr = MathUtils.eqm(w, x, d)
            print('epoch = {} w = {} eqm = {}'.format(epoch, w, eqm_curr - eqm_prev))
            if abs(eqm_curr - eqm_prev) < e:
                break
        return w
            
    def test(self, w, x):
        v = np.dot(np.transpose(w), x)
        y = MathUtils.step(v)
        return y;

if  __name__ == '__main__':
    
    # data
    x = SampleData.add_bias(SampleData.OR.input, -1)
    d = SampleData.OR.output
    
    # neural network
    nn = SingleLayerPerceptronDelta()
    
    # train
    w = nn.train(x, d, 1e-3, 1e-9)
    
    # test
    correct = 0
    for i in range(0,len(x)):
        y = nn.test(w, x[i])
        if (y == d[i]):
            correct = correct + 1
    print('accuracy: {}/{} ({}%)'.format(correct, len(x), 100.0 * float(correct) / float(len(x))))
