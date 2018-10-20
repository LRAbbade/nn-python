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

    def __init__(self,x,d):
        
        self.h1 = len(x[0])
        self.h2 = min(len(x[0]/2),1)
        self.n = 1e-4 # learning rate
        self.e = 1e-7 # error threshold
        self.g = MathUtils.tanh # activation function
        self.g_d = MathUtils.tanh_d # activation function derivative
        self.plot_data_x = [] # epochs for plotting
        self.plot_data_y = [] # eqms for plotting

    def eqm(self, w, x, d):
        
        k = len(x)
        ans = 0
        
        for j in range(0, k):
            i,y = self.feed_forward(w, x[j])
            ans = ans + (d[j] - y[2])**2
            ans = ans / 2
        ans = ans / k
        
        return ans

    def feed_forward(self, w, x_i):
        
        i = np.asarray([np.zeros([self.h1,1]), np.zeros([self.h2,1]), np.zeros([1,1])])
        y = np.asarray([np.zeros([self.h1+1,1]), np.zeros([self.h2+1,1]), np.zeros([1,1])])
        
        # feed from input layer to first hidden layer
        i[0] = np.dot(w[0], x_i)
        y[0] = self.g(i[0])
        y[0] = np.insert(y[0], 0, -1.0)
        
        # feed from first hidden layer to last hidden layer
        i[1] = [None] * self.h2
        for k in range(0, self.h2):
            i[1][k] = np.asarray(np.dot(np.transpose(w[1][k]), y[0]))
        y[1] = self.g(i[1])
        y[1] = np.insert(y[1], 0, -1.0)
        
        # feed from last hidden layer to output layer
        i[2] = np.dot(np.transpose(w[2]), y[1])
        y[2] = self.g(i[2])
        
        return i , y

    def back_propagate(self, x_i, d_i, i, y, w):
        
        delta = [None] * 3
        
        # propagate from output layer to last hidden layer
        delta[2] = np.subtract(d_i, y[2]) * self.g_d(i[2])
        w[2] = w[2] + np.multiply(np.multiply(self.n, delta[2]), y[1])
        
        # propagate from last hidden layer to first hidden layer
        delta[1] = 0
        for k in range(0, len(w[2])):
            delta[1] += np.dot(delta[2], w[2][k]) 
        delta[1] = np.multiply(delta[1], self.g_d(i[1]))
        for k in range(0, len(delta[1])):
            w[1] = w[1] + np.multiply(np.multiply(self.n, delta[1][k]), y[0])
        
        # propagate from first hidden layer to input layer
        delta[0] = 0
        for k in range(0, len(w[1])):
            delta[0] += np.dot(delta[1][k], w[1][k])
        adjust = np.zeros([len(delta[0]),len(w[0][0])])
        for k in range(0, len(delta[0])):
            adjust[k] = np.multiply(delta[0][k], self.g_d(i[0]))
            w[0] = w[0] + np.multiply(np.multiply(self.n, adjust[k]), x_i)
        
        return w

    def train(self, x, d):
        
        # number of samples
        k = len(x)
        
        # randomly initialize synaptic weights
        w0 = np.random.uniform(-1.0, +1.0, (self.h1, len(x[0])))
        w1 = np.random.uniform(-1.0, +1.0, (self.h2, self.h1+1))
        w2 = np.random.uniform(-1.0, +1.0, (self.h2+1))
        w = [w0, w1, w2]
        
        # initialize epoch counter
        epoch = 0
        
        # train until eqm < e
        while True:

            # eqm before weight adjust 
            eqm_prev = self.eqm(w, x, d)
            
            # present all samples
            for j in range(0, k):
                # forward step
                i,y = self.feed_forward(w, x[j])
                # backward step
                w = self.back_propagate(x[j], d[j], i, y, w)
            
            # eqm after weight adjust                
            eqm_curr = self.eqm(w, x, d)

            # eqm absolute delta
            eqm_delta = abs(eqm_curr - eqm_prev)
            
            # increment epoch counter
            epoch = epoch + 1
            
            # print debug line and add plot data
            print('epoch = {}\terror = {}\tdelta = {}'.format(epoch, eqm_curr, eqm_delta))
            
            # append plot data
            self.plot_data_x.append(epoch)
            self.plot_data_y.append(eqm_curr)
            
            # stop condition
            if eqm_delta < self.e:
                break
            
        return w
    
    def test(self, w, x):
        
        i,y = self.feed_forward(w, x)
        
        return y;

if  __name__ == '__main__':
    
    # set random number generator seed
    np.random.seed(DataUtils.random_seed())
    
    # set floating point formatting when printing
    np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

    # load data
    x = SampleData.DIABETES.input
    d = SampleData.DIABETES.output
    
    # prepare data
    x = DataUtils.add_bias(x)
    x,d = DataUtils.shuffle(x,d)
    x_train,x_test = DataUtils.splitTrainTest(x)
    d_train,d_test = DataUtils.splitTrainTest(d)
    
    # create the neural network
    nn = MultilayerPerceptron(x_train, d_train)
    
    # train the neural network
    w = nn.train(x_train, d_train)
    
    # plot epoch versus eqm data
    PlotUtils.plot(nn.plot_data_x, 'epoch', nn.plot_data_y, 'error')
    
    # test the neural network
    correct = 0
    for i in range(0, len(x_test)):
        y = nn.test(w, x_test[i])
        if ((y[2] < 0 and d_test[i] == -1) or (y[2] > 0 and d_test[i] == +1)):
            correct = correct + 1
    accuracy = 100.0 * float(correct) / float(len(x_test))
    print('accuracy: {}/{} ({:.2f}%)'.format(correct, len(x_test), accuracy))
