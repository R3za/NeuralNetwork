from Exceptions.error import *
from regularizer import Regularizer


class L2_reg(Regularizer):

    def __init__(self,decay_weights):
        if not isinstance(decay_weights,float):
            raise ParamTypeError('L2_Regularizer','weight decay','float')

        self.weight_decay=decay_weights
        Regularizer.__init__(self,'L2_Regularizer')


    def get_gradient(self,**kwargs):
        '''
        it return the graident w.r.t. l2 reg for given weights
        :param kwargs:
        :return:
        '''
        # for k in kwargs.keys():
        #     print k

        if kwargs['W'] is None:
            raise KeyNotExistError('Parameter of L2_reg','W')
        # print self.weight_decay
        # print type(self.weight_decay)
        # x=
        # print x
        return {'W':self.weight_decay*kwargs['W']}

    def get_weight_decay(self): return self.weight_decay