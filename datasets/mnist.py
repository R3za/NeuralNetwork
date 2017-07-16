import numpy as np
import pickle as pkl
from dataset import Data_set
'''
MNIST class load the mnist data set and provides
methods to access train and test and validation data sets
'''
class MNIST(Data_set):
    def __init__(self,file_path):
        Data_set.__init__(self,file_path=file_path,name='MNIST')


            # print self.valid_set[0].shape[0]


    def load(self,file_path):
        '''
        Converts the label from digit to One_hot vector and separate label and images for test and train
        :param train_set:
        :param valid_set:
        :param test_set:
        :return:
        '''
        with open(file_path,'rb') as f:
            '''
                Each set is a tuple (Image,Label)
                Image is batch_size by 784 matrix
                Label is batch_size by 1 --> digit
                train_set image size is (50000,784)
                test_set image size is (10000,784)
                validation_set size is (10000,784)
            '''
            train_set, valid_set, test_set = pkl.load(f)
            self.train_set_size=train_set[0].shape[0]
            self.test_set_size=test_set[0].shape[0]
            self.valid_set_size=valid_set[0].shape[0]
            # self.one_hot_conversion_and_separation(train_set, valid_set, test_set)

            f.close()
            self.train_label=np.zeros((self.train_set_size,10))
            x=[i for i in range(0,self.train_set_size)]
            self.train_label[np.array(x),train_set[1]]=1
            self.train_set_images=train_set[0]

            self.valid_label=np.zeros((self.valid_set_size,10))
            x=[i for i in range(0,self.valid_set_size)]
            self.valid_label[np.array(x),valid_set[1]]=1
            self.valid_set_images=valid_set[0]

            self.test_label = np.zeros((self.test_set_size, 10))
            x = [i for i in range(0, self.test_set_size)]
            self.test_label[np.array(x), test_set[1]] = 1
            self.test_set_images = test_set[0]



    def get_test_set_size(self): return self.test_set_size
    def get_train_set_size(self): return self.train_set_size
    def get_valid_set_size(self): return self.valid_set_size

    def get_train_data(self): return self.train_set_images.T,self.train_label.T
    def get_test_data(self): return self.test_set_images.T,self.test_label.T
    def get_valid_data(self): return self.valid_set_images.T,self.valid_label.T


    def get_next_batch(self,mini_batch_size=50):
        '''
        Picks #mini_batch_size randomly and pick them and return the corresponding images and labels
        :param mini_batch_size: the mini batch size
        :return:
        '''
        if mini_batch_size > self.train_set_size: return self.get_train_data()
        rind=np.random.randint(0,self.get_train_set_size(),mini_batch_size)
        # print rind
        return self.train_set_images[rind,:].T,self.train_label[rind,:].T

























#
# ds=MNIST('../ds_files/mnist.pkl')
# x,y=ds.get_next_batch(mini_batch_size=50)
# print x.shape
# print y.shape