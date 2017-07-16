import numpy as np
from activation import Activation

class Relu(Activation):

    @staticmethod
    def get_instance(activation_name,scope):
        return Relu(scope)

    def __init__(self,scope):
        Activation.__init__(self,scope, 'Relu')

    def fire(self,Y):
        '''
        Apply the Activations funciton on the input
        :param Y: input matrix or tensor of size (d,m,n,...) d --> dimension, m,n,... the rest dimension for
                                                            mini batch or filter or chanels or ...
        :return: matrix of size Y
        '''
        self.inputcheck(Y,np.ndarray,'numpy ndarray')
        return np.maximum(Y,0)


    def gradient(self,Y,backprop):
        '''
        Computing the gradient of relu w.r.t. Y. Note the graident is either 0 or 1
        :param Y: input matrix or tensor
        :return:
        '''
        return backprop * np.sign(self.fire(Y))





#
# rel=Relu('Test')
# rel.fire(None)