from Losses.loss import *
from Losses.cross_entropy import *
from ann import NN
from Minimizers import sgd,lrsched
import time

'''
    This class gets a neural net and loss function and
    an optimization algorithm and run training
'''

class Trainer:

    def __init__(self,ann=NN(),loss=Loss,opt_alg=sgd.SGD,lr_sched=lrsched.ConstantLrSched):
        self.name='Trainer'
        self.ann=ann
        self.loss=loss
        self.opt_alg=opt_alg
        self.lrschd=lr_sched
        self.train_accuracy=[]
        self.valid_accuracy=[]
        self.test_accuracy=[]

    def trian(self,data_set,num_epoch,mini_batch_size,
              train_accuracy_report_period=0,
              valid_accuracy_report_period=0,test_accuracy_report_period=0):
        dss=data_set.get_train_set_size()
        if mini_batch_size > dss:
            mini_batch_size=dss

        for epoch in range(num_epoch):
            btime=time.time()
            print 'epoch :',epoch,
            for itrpe in range(np.int32(dss/mini_batch_size)):
                # Getting batch of data
                X,Y=data_set.get_next_batch(mini_batch_size=mini_batch_size)

                # do feedforward and compute the network output
                y_nn=self.ann.feedforward(X)
                # compute the graident of loss function w.r.t. to the output of network
                Ly = self.loss.get_gradient(Y,y_nn)
                # start backprop in the network with the graident vector w.r.t. network output
                self.ann.backProp(Ly)

                # Could be parallelized
                for key in self.ann.layers.keys():
                    layer=self.ann.layers[key]
                    self.opt_alg.minimize(layer.get_param_grad_dict(),self.lrschd.get_lr(epoch*dss+itrpe))
                    # layer.update_param()

            self.acuracy_report(train_accuracy_report_period,X,Y,epoch,self.train_accuracy)

            self.acuracy_report(test_accuracy_report_period,(data_set.get_test_data())[0]
                                ,(data_set.get_test_data())[1],epoch,self.test_accuracy)

            self.acuracy_report(valid_accuracy_report_period,(data_set.get_valid_data())[0]
                                ,(data_set.get_valid_data())[1],epoch,self.valid_accuracy)

            etime=time.time()
            print 'length=',etime-btime

        return self.train_accuracy,self.valid_accuracy,self.test_accuracy

    def acuracy_report(self,period,data_set_x,data_set_label,epoch,report_list):
        if period is not 0:
            if (epoch+1) % period == 0:
                y_nn=self.ann.feedforward(data_set_x)
                report_list.append(self.loss.get_accuracy(data_set_label,y_nn))



class TestTrainer:
    def __init__(self,ann=NN(),loss=Loss,opt_alg=sgd.SGD,lr_sched=lrsched.ConstantLrSched):
        self.name='Trainer'
        self.ann=ann
        self.loss=loss
        self.opt_alg=opt_alg
        self.lrschd=lr_sched
        self.train_accuracy=[]
        self.valid_accuracy=[]
        self.test_accuracy=[]

    def trian(self,data_set,num_itr,mini_batch_size):
        dss=data_set.get_train_set_size()
        if mini_batch_size > dss:
            mini_batch_size=dss

        for epoch in range(num_itr):
            print 'itr :',epoch
            # Getting batch of data
            X,Y=data_set.get_next_batch(mini_batch_size=mini_batch_size)

            # print 'X'
            # print X

            y_nn=self.ann.feedforward(X)
            # compute the graident of loss function w.r.t. to the output of network


            # print 'y_nn is:'
            # print y_nn

            # for key in self.ann.layers.keys():
            #     layer=self.ann.layers[key]
            #     layer.printParam()

            Ly = self.loss.get_gradient(Y,y_nn)

            # print 'loss of label'
            # print Ly
            # start backprop in the network with the graident vector w.r.t. network output
            self.ann.backProp(Ly)

            # Could be parallelized
            for key in self.ann.layers.keys():
                layer=self.ann.layers[key]
                self.opt_alg.minimize(layer.get_param_grad_dict(),self.lrschd.get_lr(epoch))
                # layer.printParam()


