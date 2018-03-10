'''
Created on 5 de dez de 2017

@author: marcelovca90
'''
import numpy as np
import MathUtils
import PlotUtils
import SampleData

class SingleLayerPerceptronHebbian:

    def __init__(self):
        print('SingleLayerPerceptronHebbian')

    def train(self, x, d, n):
        plot_data_x = []
        plot_data_y = []
        k = len(x)
        w = np.random.rand(len(x[0]))
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
            plot_data_x.append(epoch)
            plot_data_y.append(error)
        PlotUtils.plot(plot_data_x, 'epoch', plot_data_y, 'error')
        return w
            
    def test(self, w, x):
        v = np.dot(np.transpose(w), x)
        y = MathUtils.step(v)
        return y;


if  __name__ == '__main__':
        
    # read data
    x = MathUtils.add_bias(SampleData.TIC_TAC_TOE_ENDGAME.input, -1)
    d = SampleData.TIC_TAC_TOE_ENDGAME.output
    
    # prepare data
    x,d = MathUtils.shuffle(x,d)
    x_train,x_test = MathUtils.split(x)
    d_train,d_test = MathUtils.split(d)
    
    # neural network
    nn = SingleLayerPerceptronHebbian()
    
    # train
    n = 1e-1  # learning rate
    w = nn.train(x_train, d_train, n)
    
    # test
    correct = 0
    for i in range(0, len(x_test)):
        y = nn.test(w, x_test[i])
        if (y == d_test[i]):
            correct = correct + 1
    accuracy = 100.0 * float(correct) / float(len(x_test))
    print('accuracy: {}/{} ({:.2f}%)'.format(correct, len(x_test), accuracy))
