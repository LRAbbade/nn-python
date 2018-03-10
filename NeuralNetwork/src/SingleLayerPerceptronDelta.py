'''
Created on 5 de dez de 2017

@author: marcelovca90
'''
import numpy as np
import MathUtils
import SampleData
import PlotUtils

class SingleLayerPerceptronDelta:

    def __init__(self):
        print('SingleLayerPerceptronDelta')

    def train(self, x, d, n, e):
        plot_data_x = []
        plot_data_y = []
        k = len(x)
        w = np.random.rand(len(x[0]))
        epoch = 0
        while True:
            eqm_prev = MathUtils.eqm(w, x, d)
            for i in range(0, k):
                v = np.dot(np.transpose(w), x[i])
                w = np.add(w, np.multiply(x[i], n * (d[i] - v)))
            epoch = epoch + 1
            eqm_curr = MathUtils.eqm(w, x, d)
            eqm_delta = abs(eqm_curr - eqm_prev)
            print('epoch = {}\tw = {}\teqm = {}'.format(epoch, w, eqm_delta))
            plot_data_x.append(epoch)
            plot_data_y.append(eqm_delta)
            if eqm_delta < e:
                break
        PlotUtils.plot(plot_data_x, 'epoch', plot_data_y, 'eqm')
        return w
            
    def test(self, w, x):
        v = np.dot(np.transpose(w), x)
        y = MathUtils.step(v)
        return y;

if  __name__ == '__main__':
    
    # data
    x = MathUtils.add_bias(SampleData.TIC_TAC_TOE_ENDGAME.input, -1)
    d = SampleData.TIC_TAC_TOE_ENDGAME.output
    
    # neural network
    nn = SingleLayerPerceptronDelta()
    
    # train
    n = 1e-1 # learning rate
    e = 1e-3 # error threshold
    w = nn.train(x, d, n, e)
    
    # test
    correct = 0
    for i in range(0, len(x)):
        y = nn.test(w, x[i])
        if (y == d[i]):
            correct = correct + 1
    print('accuracy: {}/{} ({}%)'.format(correct, len(x), 100.0 * float(correct) / float(len(x))))
