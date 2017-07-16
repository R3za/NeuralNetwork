
class Layer:

    def __init__(self,scope=None,name='general_layer'):

        self.scope=str(scope)
        self.name=str(scope)+'_'+str(name)
        '''
        list containing regualizers for this layer
        '''
        self.regList=[]
        '''
        This dictionary contains all the variables and parameters needed for a layer
        Each element is a list which has 3 elements in it, the parameter, its graideint and a flag
        to see if it is trainable or not. if it is not trainable its gradient is None
        '''
        self.param_grad_dict = {}

        # this is for test mode printing in layer functions
        self.testmode=False

    def add_reg(self,reg):
        self.regList.append(reg)
    '''
        return the output of this layer after applying Activations
    '''
    def feedForward(self,X,**kwargs):
        return None

    '''
        return the drivitive of this layer w.r.t variables and biases
    '''
    def backprop(self,backPropMatrix,**kwargs):
        return None

    def giveOutput(self,input):
        '''
        return the output of this layer after applying Activations
        :param input: input matrix or tensor
        :return:
        '''
        return input

    def set_next_layer(self,layer):
        self.next_layer=layer

    def get_param_grad_dict(self): return None

    def printParam(self): print self.param_grad_dict
'''
A centiniel layer for the last layer feedforward
'''
class Identity(Layer):

    def feedForward(self,X,**kwargs):
        return X


class NoneLayer(Layer):
    def __init__(self):
        self.scope='None'
        self.name='_'+'None'

