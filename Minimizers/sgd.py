import numpy as np
from minimizer import *

class SGD(Minimizer):

    def __init__(self):
        Minimizer.__init__(self,'SGD')

    def minimize(self,var_grad_dict,lr):
        '''
        Do the SGD update
        :param var: variable needs to be updated
        :param grad: the gradient of objective function w.r.t. var
        :param lr: learning rate
        :return:
        '''
        for k in var_grad_dict.keys():
            if var_grad_dict[k][2]:
                (var_grad_dict[k])[0]=(var_grad_dict[k])[0]-lr*((var_grad_dict[k])[1])
