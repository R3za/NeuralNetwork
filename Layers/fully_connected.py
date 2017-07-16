import numpy as np

from Activations import relu
from Exceptions import error
from Regularizers import dropout
from layer import *


class Fc_Layer(Layer):

    def __init__(self, scope, fan_in, fan_out,
                 activation_function=relu.Relu('Default'),
                 prev_layer=NoneLayer(),keep_prob=1.0):
        '''
        :param scope: scope of this layer in the arch. of NN like layer3 or l3 or fc3
        :param dimension: the dimension of weights
        :param size: # of weights
        :param activation_function: Activations function, default Relu
        :param prev_layer: the previous layer this layer is connected too, could be NULL if this is the first layer
        '''

        Layer.__init__(self,scope,'Fully_connected')

        self.actv_func=activation_function
        self.fan_out=fan_out
        self.fan_in=fan_in

        self.prev_layer=prev_layer
        self.next_layer=Identity()


        '''
        This dictionary contains all the needed parameters and thier graidents for this layer
        Each element is a list: the first element is the parameter value
                                the second is parameter gradient
                                the thired is a flag saying if this parameter is trainable or not
        '''
        self.param_grad_dict = {'W':[np.random.normal(np.float64(0),0.01,(fan_out,fan_in),),np.zeros((fan_out,fan_in)),True],
                                'b':[np.zeros((fan_out,1)),np.zeros((fan_out,1)),True],
                                'X':[None,None,False]}

        self.dropout = None
        self.keep_prob=keep_prob
        if keep_prob is not 1.0:
            self.dropout=dropout.Dropout(keep_prob=keep_prob,shape=(fan_out,1))



    def getW(self): return self.param_grad_dict['W'][0]
    def getdW(self): return self.param_grad_dict['W'][1]
    def getb(self): return self.param_grad_dict['b'][0]
    def getdb(self): return self.param_grad_dict['b'][1]
    def getX(self): return self.param_grad_dict['X'][0]

    def setW(self,newW): self.param_grad_dict['W'][0]=newW
    def setdW(self,newdW): self.param_grad_dict['W'][1]=newdW
    def setb(self,newb): self.param_grad_dict['b'][0]=newb
    def setdb(self,newdb): self.param_grad_dict['b'][1]=newdb
    def setX(self,newX): self.param_grad_dict['X'][0]=newX





    def setKeep_prob(self,keep_prob):
        '''
        For test time keep prob should be set to 1 again
        :param keep_prob:
        :return:
        '''
        self.keep_prob=keep_prob

    def feedForward(self,X,**kwargs):
        '''
        Compute the output of this layer and forward it to next layer
        :param X: datasets matrix which is the form (d,m): d --> dimension and m --> mini_batch size
        if X does not have the required structure this layer reshape it to required format
        :return: if it is the last layer it returns this layer's output otherwise it calls the output of next layer
        '''
        if X is None:
            raise error.LayerInputError(self.name)

        if self.dropout is not None:
            self.dropout.update_gates()

        self.setX(X)
        if self.testmode:
            print self.name
            print 'FeedForward'
            print 'W_shape'
            print self.getW().shape
            print'Input shape'
            print X.shape
            print 'linear transform in feedforward'

            print np.add(np.dot(self.getW(),X),self.getb())

        return self.next_layer.feedForward(self.actv_func.fire(np.add(np.dot(self.getW(),X),self.getb())))


    def giveOutput(self, input):
        self.actv_func.fire(np.add(np.dot(self.getW(), input),self.getb()))

    def get_weight(self):return self.getW()
    def get_bias(self):return self.getb()

    def backprop(self,backPropMatrix,**kwargs):
        '''
        This operation compute the backprop operation through this layer
        it compute gradient w.r.t. W and b and X
        It saves graidents of W and b but passes griadient w.r.t. X to previous layer if it is not null
        :param backPropMatrix: The chain rule result from above layers or right layers :)
        :param kwargs:
        :return: calls previous layer backprop to compute its graidient
        '''

        if self.getX() is None:
            raise error.LayerInputError(self.name)

        # gradient of the output of the active function w.r.t. affine transform multiply to backprop
        # dimension of Lz is (fan_out,m): fan_out: number of output of this layer and m is mini_batch
        Lz=self.actv_func.gradient(np.add(np.dot(self.getW(), self.getX()) , self.getb()),backPropMatrix)
        d,m=Lz.shape

        # dw is matrix of (fan_out, fan_in)
        self.setdW(np.dot(Lz,self.getX().T)/np.float32(m))
        self.setdb(np.sum(Lz,axis=1).reshape((Lz.shape[0],1)))

        if self.testmode:
            print self.name
            print 'Lz'
            print Lz
            print 'W'
            print self.getW()
            print 'b'
            print self.getb()
            print 'dw'
            print self.getdW()
            print 'db'
            print self.getdb()

        for reg in self.regList:
            reg_grad_dict=reg.get_gradient(**{'W':self.getW(),'b':self.getb()})
            for k in reg_grad_dict.keys():
                (self.param_grad_dict[k])[1]=(self.param_grad_dict[k])[1]+reg_grad_dict[k]

        self.prev_layer.backprop(np.dot(self.getW().T,Lz))

    def get_param_grad_dict(self): return self.param_grad_dict

    def printParam(self):
        print 'W,dw\n',self.param_grad_dict['W']


    # def update_param(self):
    #     '''
    #     Should be called to update param after minimization operation
    #     :return:
    #     '''
    #     self.W=self.grad_dict['W'][0]
    #     self.b = self.grad_dict['b'][0]
    #


