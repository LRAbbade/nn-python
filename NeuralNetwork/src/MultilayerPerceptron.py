'''
Created on 16 de mar de 2018

@author: marcelovca90
'''
import numpy as np
from util import DataUtils
from util import MathUtils
from util import PlotUtils
from data import SampleData

class MultilayerPerceptron:

    def __init__(self):
        self.n = 1e-3 # learning rate
        self.e = 1e-3 # error threshold
        self.l = 4 # neurons in the hidden layer
        self.g = MathUtils.tanh # activation function
        self.g_d = MathUtils.tanh_d # activation function derivative
        self.plot_data_x = [] # epochs for plotting
        self.plot_data_y = [] # eqms for plotting

    def train(self, x, d):
        k = len(x)
        w = np.random.rand(3, len(x[0]))
        epoch = 0
        while True:
            eqm_prev = MathUtils.eqm(w, x, d)
            for i in range(0, k):

                # forward step
                
                i_1 = np.dot(np.transpose(w[0]), x[i])
                y_1 = self.g(i_1)
                
                i_2 = np.dot(np.transpose(w[1]), y_1)
                y_2 = self.g(i_2)
                
                i_3 = np.dot(np.transpose(w[2]), y_2)
                y_3 = self.g(i_3)
                
                # backward step
                
                delta_3 = np.subtract(d[i], y_3) * self.g_d(i_3)
                w[2] = w[2] + np.multiply(np.multiply(self.n, delta_3), y_2)
                
                delta_2 = np.dot(delta_3, w[2]) * self.g_d(i_2)
                w[1] = w[1] + np.multiply(np.multiply(self.n, delta_2), y_1)
                
                delta_1 = np.dot(delta_2, w[1]) * self.g_d(i_1)
                w[0] = w[0] + np.multiply(np.multiply(self.n, delta_1), x[i])
                            
            eqm_curr = MathUtils.eqm(w, x, d)
            eqm_delta = sum(abs(np.subtract(eqm_curr,eqm_prev)))
            epoch = epoch + 1
            print('epoch = {}\tw = {}\teqm(abs) = {}'.format(epoch, w, eqm_delta))
            self.plot_data_x.append(epoch)
            self.plot_data_y.append(eqm_delta)
            if np.all(eqm_delta < self.e):
                break
        return w
            
    def test(self, w, x):
        v = np.dot(np.transpose(w), x)
        y = self.g(v)
        return y;

if  __name__ == '__main__':
    
    # load data
    x = SampleData.XOR_GATE.input
    d = SampleData.XOR_GATE.output
    
    # prepare data
    x = DataUtils.add_bias(x)
    x,d = DataUtils.shuffle(x,d)
    x_train,x_test = DataUtils.split(x)
    d_train,d_test = DataUtils.split(d)
    
    # create the neural network
    nn = MultilayerPerceptron()
    
    # train the neural network
    w = nn.train(x_train, d_train)
    
    # plot epoch versus eqm data
    PlotUtils.plot(nn.plot_data_x, 'epoch', nn.plot_data_y, 'eqm(abs)', nn.e)
    
    # test the neural network
    correct = 0
    for i in range(0, len(x_test)):
        y = nn.test(w, x_test[i])
        if ((y[2] < 0 and d_test[i] == -1) or (y[2] > 0 and d_test[i] == +1)):
            correct = correct + 1
    accuracy = 100.0 * float(correct) / float(len(x_test))
    print('accuracy: {}/{} ({:.2f}%)'.format(correct, len(x_test), accuracy))
