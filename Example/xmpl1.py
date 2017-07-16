from ANNs.ann import NN
from Regularizers.l2_reg import *
from ANNs.trainer import *
from Losses.cross_entropy import *
from Minimizers.sgd import *
from Minimizers.lrsched import *
from datasets.mnist import *
from datasets.dataset import *

ann=NN('TEST_ANN_For_MNIST',testmode=False)

ann.addFCLayer(fan_in_size=784,fan_out_size=1024,activation_name='relu')
ann.addFCLayer(fan_out_size=512,activation_name='relu')
ann.addFCLayer(fan_out_size=256,activation_name='relu')
ann.addFCLayer(fan_out_size=10,activation_name='softmax')

ann.add_layerwise_Regularizer(reg=L2_reg(0.00001))

loss=Cross_Entropy()
opt_alg=SGD()
# lr_schd=Grad_Reduce_LrSched(red_coef=0.5,red_period=1000,initial_lr=5e-2)
lr_schd=ConstantLrSched(initial_learning_rate=1e-2)
trainer=Trainer(ann=ann,loss=loss,opt_alg=opt_alg,lr_sched=lr_schd)

data_set=MNIST('../ds_files/mnist.pkl')


X_test,Y_test=data_set.get_test_data()
Y_nn=ann.feedforward(X_test)
print 'before training:', loss.get_accuracy(Y_test,y_nn=Y_nn)

test_period=10
lists=trainer.trian(data_set=data_set,num_epoch=100,mini_batch_size=150,train_accuracy_report_period=test_period,
                    test_accuracy_report_period=test_period,valid_accuracy_report_period=test_period)

X_test,Y_test=data_set.get_test_data()
Y_nn=ann.feedforward(X_test)
print 'after training:',loss.get_accuracy(Y_test,y_nn=Y_nn)


print lists