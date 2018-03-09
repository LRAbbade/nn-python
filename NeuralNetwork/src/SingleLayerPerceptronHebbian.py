'''
Created on 5 de dez de 2017

@author: marcelovca90
'''
import numpy as np
import MathUtils
import SampleData

class SingleLayerPerceptronHebbian:

    def __init__(self):
        print('SingleLayerPerceptronHebbian')

    def train(self, x, d, n):
        k = len(x)
        w = np.random.rand(k-1)
        epoch = 0
        error = True
        while error:
            error = False
            for i in range(0, k):
                v = np.dot(np.transpose(w), x[i])
                y = MathUtils.step(v)
                if y != d[i]:
                    w = np.add(w, np.multiply(x[i], n * (d[i] - y)))
                    error = True
            epoch = epoch + 1
            print('epoch = {}\tw = {}\terror={}'.format(epoch, w, error))
        return w
            
    def test(self, w, x):
        v = np.dot(np.transpose(w), x)
        y = MathUtils.step(v)
        return y;


if  __name__ == '__main__':
    
    # data
    x = MathUtils.add_bias(SampleData.OR.input, -1)
    d = SampleData.OR.output
    
    # neural network
    nn = SingleLayerPerceptronHebbian()
    
    # train
    n = 1e-1  # learning rate
    w = nn.train(x, d, n)
    
    # test
    correct = 0
    for i in range(0, len(x)):
        y = nn.test(w, x[i])
        if (y == d[i]):
            correct = correct + 1
    print('accuracy: {}/{} ({}%)'.format(correct, len(x), 100.0 * float(correct) / float(len(x))))
