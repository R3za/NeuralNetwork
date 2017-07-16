import numpy as np

class Dropout:
    '''
    This class implement the basic idea of dropout
    '''
    def __init__(self,shape,keep_prob=0.5):
        '''
        Initiate a dropout setting for a given layer
        :param shape: the shape of binary vector
        :param keep_prob: the keeping probability or probability of being one
        '''
        self.keep_prob=keep_prob
        self.shape=shape
        self.gate_vector=None

    def get_gates(self):
        return self.gate_vector

    def update_gates(self):
        self.gate_vector=np.sign(np.maximum(np.random.uniform(low=0,high=1,size=self.shape)-self.keep_prob,0))
