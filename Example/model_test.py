from ANNs.ann import NN
from Regularizers.l2_reg import *
from ANNs.trainer import *
from Losses.cross_entropy import *
from Minimizers.sgd import *
from Minimizers.lrsched import *
from datasets.mnist import *
from datasets.dataset import *

ann=NN('TEST_ANN_For_Test_data',testmode=False)

ann.addFCLayer(fan_in_size=5,fan_out_size=5,activation_name='softmax')

# ann.add_layerwise_Regularizer(reg=L2_reg(0.0001))

loss=Cross_Entropy()
opt_alg=SGD()

lr_schd=ConstantLrSched(constant_learning_rate=1e-4)
trainer=TestTrainer(ann=ann,loss=loss,opt_alg=opt_alg,lr_sched=lr_schd)
# data_set=MNIST('../ds_files/mnist.pkl')
data_set=Test_Data_set()


test_period=0
X_test,Y_test=data_set.get_train_data()
Y_nn=ann.feedforward(X_test)
print 'before training:', loss.get_accuracy(Y_test,y_nn=Y_nn)

trainer.trian(data_set=data_set,num_itr=20000,mini_batch_size=1)

X_test,Y_test=data_set.get_train_data()
Y_nn=ann.feedforward(X_test)
print 'after training:', loss.get_accuracy(Y_test,y_nn=Y_nn)

