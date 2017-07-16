class Data_set:
    def __init__(self,file_path,name='Data Set'):
        self.name=name
        self.load(file_path)

    def load(self,directory): return None
    def get_test_set_size(self): return 0
    def get_train_set_size(self): return 0
    def get_valid_set_size(self): return 0
    def get_train_data(self): return None
    def get_test_data(self): return None
    def get_valid_data(self): return None

    def get_next_batch(self, mini_batch_size=50): return None

import numpy as np

class Test_Data_set(Data_set):

    def __init__(self, name='Test Data Set'):
        self.name = name
        self.train_set_images, self.train_label = np.array(
            [[1, 1, 1, 0, 0], [1, 2, 1, 0, 0], [1, 0, 3, 0, 0], [1,0,0,4,0], [1, 0, 0, 0, 5]]), \
                                                  np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0],
                                                            [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    def get_test_set_size(self): return 3
    def get_train_set_size(self): return 5

    def get_valid_set_size(self): return 2

    def get_train_data(self):

        return self.train_set_images,self.train_label

    # def get_test_data(self): return np.array([[1,1,1, 1 ,1],[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4],[5,5,5,5,5]]),\
    #            np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])

    def get_valid_data(self): return None

    def get_next_batch(self, mini_batch_size=50):
        rind=np.random.randint(0,self.get_train_set_size(),mini_batch_size)
        # print rind
        return self.train_set_images[rind,:].T,self.train_label[rind,:].T
