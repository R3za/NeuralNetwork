import numpy as np
from activation import Activation

class Softmax(Activation):

    @staticmethod
    def get_instance(activation_name, scope):
        return Softmax(scope)

    def __init__(self,scope):
        Activation.__init__(self,scope,'Softmax')

    def fire(self,Y):
        '''
        Compute the softmax for Y. Here we can use logsumexp trick to prevent overflow
        :param Y: is (d,m) Matrix and m is mini_batch size.
        :return: softmax(Y)
        '''
        self.inputcheck(Y,np.ndarray,'numpy ndarray')
        Ymax=np.max(Y,axis=0)

        return np.exp(Y-Ymax-np.log(np.sum(np.exp(Y-Ymax),axis=0)))

    def gradient(self,Y,backprop):
        '''
        :param Y: matrix of (d,m)- d: # of output, m: mini_batch size
        :param backprop: back_prop from previous layer
        :return: gradient w.r.t. these ouput layers
        '''
        # a=self.simoid(Y)
        # return a*(1-a)
        pz=self.fire(Y)
        # print 'pz is\n',pz
        # print 'backprop from loss\n',backprop
        # print 'gradient\n',pz*(backprop-np.sum(backprop*pz,axis=0))
        return pz*(backprop-np.sum(backprop*pz,axis=0))
