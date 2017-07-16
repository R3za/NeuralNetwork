
from Exceptions.error import *

class Loss:


    @staticmethod
    def get_instance(loss_name):
        from cross_entropy import *
        reg_loss = {'cross_entropy': Cross_Entropy}
        if Loss.reg_loss[loss_name] is None:
            raise KeyNotExistError('loss list','loss name')

        return Loss.reg_loss[loss_name].get_instance()

    def __init__(self,name='General Losses'):
        self.name=name
        self.testmode=False

    def compute_loss(self, y_true, y_nn):
        return None

    def get_gradient(self,y_true,y_nn):
        return None

    def get_accuracy(self,y_true,y_nn):
        return None
