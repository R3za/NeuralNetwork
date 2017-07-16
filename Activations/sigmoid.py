import numpy as np
from Activations.activation import Activation

class Sigmoid(Activation):

    @staticmethod
    def get_instance(activation_name, scope):
        return Sigmoid(scope)


    def __init__(self,scope):
        Activation.__init__(self,scope,'Sigmoid')


    def fire(self,Y):
        self.inputcheck(Y,np.ndarray,'numpy ndarray')
        return 1.0/(1+np.exp(-Y))


    def gradient(self,Y,backprop):
        a=self.simoid(Y)
        return a*(1-a)
