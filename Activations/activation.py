from Exceptions.error import *


class Activation(object):


    @staticmethod
    def reg_activation(activation_name,activation_class):
        Activation.reg_ac[activation_name]=activation_class

    @staticmethod
    def get_instance(activation_name,scope):
        from Activations import relu, sigmoid, softmax
        Activation.reg_ac = {'relu': relu.Relu, 'sigmoid': sigmoid.Sigmoid, 'softmax': softmax.Softmax}

        if Activation.reg_ac[activation_name] is None:
            raise KeyNotExistError('Activation List',activation_name)
        return Activation.reg_ac[activation_name].get_instance(activation_name,scope)

    def __init__(self,scope,name='general_activation_function'):
        self.scope=scope
        self.name=str(scope)+'_'+str(name)
        self.testmode=False

    def inputcheck(self,X,type,typestr):
        '''
        Check if the input to this Activations is of the required type
        :param X: input
        :param type: class of the required type
        :param typestr: the string representing the type in order to print in error message
        :return:
        '''
        if not isinstance(X,type):
            raise InputTypeError(self.scope, typestr)

    def fire(self,Y):
        '''
        Apply the Activations funciton on the input
        :param Y: input matrix or tensor of size (d,m,n,...) d --> dimension, m,n,... the rest dimension for
                                                            mini batch or filter or chanels or ...
        :return: matrix of size Y
        '''
        return None

    def gradient(self,Y,backprop):
        '''
        Computing the gradient of the Activations function w.r.t. Y.
        :param Y: input matrix or tensor
        :return: gradient w.r.t. Y
        '''
        return None