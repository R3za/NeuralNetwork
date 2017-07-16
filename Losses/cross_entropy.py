import numpy as np
from loss import Loss


class Cross_Entropy(Loss):

    @staticmethod
    def get_instance(loss_name):
        return Cross_Entropy()

    def __init__(self):
        Loss.__init__(self,'Cross_Entropy')

    def compute_loss(self,y_true_label,y_nn):
        '''
        Computing the cross entropy between y_truth and y_nn
        :param y_true_label: is one hot probability vector or matrix of size (d,m): d ---> # of labels and m --> mini bach size
        :param y_nn: probability vector of size (d,m): d ---> # of labels and m --> mini bach size
        :return: cross entropy between y_t and y_nn
        '''

        return np.sum(y_true_label*np.log(y_nn),axis=0)


    def get_gradient(self,y_true_label,y_nn):
        '''
        Computing the gradient of this loss w.r.t. y_nn variables
        :param y_true_label: is one hot probability vector or matrix of size (d,m): d ---> # of labels and m --> mini bach size
        :param y_nn: probability vector of size (d,m): d ---> # of labels and m --> mini bach size
        :return: gradient matrix or vector
        '''
        # print y_nn[:,0]
        return -(y_true_label*1.0)/y_nn

    def get_accuracy(self,y_true_label,y_nn):
        '''
        compare the output of the network with true output
        :param y_true_label: true label, matrix of size (d,m): d # of label and m is mini batch size
        :param y_nn: nn ouput : matrix of size (d,m): d # of label and m is mini batch size
        y_nn is one_hot vector in classification case
        :return: accuracy
        '''
        d,m=np.shape(y_nn)
        y_nn_max=np.int32(np.divide(np.float64(y_nn),np.max(y_nn,axis=0)))
        # print y_nn_max
        return (np.sum(y_true_label*y_nn_max)*100)/np.float32(m)
