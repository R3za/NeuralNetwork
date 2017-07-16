from Activations.activation import Activation
from Exceptions.error import *
from Layers.fully_connected import Fc_Layer as fcl
from Layers.layer import *
from Regularizers.regularizer import Regularizer


class NN:

    def __init__(self,name='ANNs',testmode=False):
        self.name=name
        self.firstLayer=NoneLayer()
        self.lastLayer=NoneLayer()
        self.layercounter=0
        self.layers={}
        self.testmode=testmode

    def addFCLayer(self,fan_in_size=None,fan_out_size=None,activation_name=None):
        '''
        Add a FC to the last layer
        :param fan_in_size: it needed just for the first layer
        :param fan_out_size:
        :param activation_name: registered name for an activation
        :return:
        '''
        if isinstance(self.firstLayer,NoneLayer):
                # print 'first layer is added'
                if fan_in_size is None:
                    raise NoneError('fan in size','first layer of NN')
                fcli=fcl(self.layercounter,fan_in_size,fan_out=fan_out_size
                         ,activation_function=Activation.get_instance(activation_name,self.layercounter))
                fcli.testmode=self.testmode

                self.lastLayer=fcli
                self.firstLayer=fcli
                self.layers[self.layercounter]=fcli

        else:
            fan_in=self.lastLayer.fan_out

            fcli = fcl(self.layercounter, fan_in, fan_out=fan_out_size
                       ,activation_function=Activation.get_instance(activation_name, self.layercounter)
                       ,prev_layer=self.lastLayer)
            fcli.testmode = self.testmode
            self.lastLayer.set_next_layer(fcli)
            self.lastLayer = fcli
            self.layers[self.layercounter] = fcli

        self.layercounter+=1

    def add_layerwise_Regularizer(self,reg=Regularizer):
        for layer in self.layers.values():
            layer.add_reg(reg)



    def feedforward(self,X):
        self.y_nn=self.firstLayer.feedForward(X)
        return self.y_nn

    # def addLoss(self,loss_name):
    #     self.loss=loss.Loss.get_instance(loss_name)

    def backProp(self,Ly):
        '''
        backProp over the network
        First it does a FF to compute y_nn and then backProp
        :param y_true: grand truth could be a matrix of size (d,m), m is mini_batch size
        :param X: input data
        :return:
        '''
        # self.feedforward(X)
        # Ly=self.loss.get_gradient(y_true,self.y_nn)
        self.lastLayer.backprop(Ly)
