'''
Created on 5 de dez de 2017

@author: marcelovca90
'''
import numpy as np
from util import MathUtils
from util import PlotUtils
from data import SampleData

class SingleLayerPerceptronDelta:

    def __init__(self):
        self.n = 1e-3 # learning rate
        self.e = 1e-3 # error threshold

    def train(self, x, d):
        plot_data_x = []
        plot_data_y = []
        k = len(x)
        w = np.random.rand(len(x[0]))
        epoch = 0
        while True:
            eqm_prev = MathUtils.eqm(w, x, d)
            for i in range(0, k):
                v = np.dot(np.transpose(w), x[i])
                w = np.add(w, np.multiply(x[i], self.n * (d[i] - v)))
            epoch = epoch + 1
            eqm_curr = MathUtils.eqm(w, x, d)
            eqm_delta = abs(eqm_curr - eqm_prev)
            print('epoch = {}\tw = {}\teqm(abs) = {}'.format(epoch, w, eqm_delta))
            plot_data_x.append(epoch)
            plot_data_y.append(eqm_delta)
            if eqm_delta < self.e:
                break
        PlotUtils.plot(plot_data_x, 'epoch', plot_data_y, 'eqm(abs)')
        return w
            
    def test(self, w, x):
        v = np.dot(np.transpose(w), x)
        y = MathUtils.step(v)
        return y;

if  __name__ == '__main__':
    
    # load data
    x = SampleData.TIC_TAC_TOE_ENDGAME.input
    d = SampleData.TIC_TAC_TOE_ENDGAME.output
    
    # prepare data
    x = MathUtils.add_bias(x,-1)
    x,d = MathUtils.shuffle(x,d)
    x_train,x_test = MathUtils.split(x)
    d_train,d_test = MathUtils.split(d)
    
    # create neural network
    nn = SingleLayerPerceptronDelta()
    
    # train the neural network
    w = nn.train(x_train, d_train)
    
    # test the neural network
    correct = 0
    for i in range(0, len(x_test)):
        y = nn.test(w, x_test[i])
        if (y == d_test[i]):
            correct = correct + 1
    accuracy = 100.0 * float(correct) / float(len(x_test))
    print('accuracy: {}/{} ({:.2f}%)'.format(correct, len(x_test), accuracy))
