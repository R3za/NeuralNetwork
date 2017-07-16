class LrSched:

    def __init__(self,name='Lr Scheduler',initial_learning_rate=1e-3):
        self.name=name
        self.lr = initial_learning_rate

    def get_lr(self,itr):
        '''
        return the correct learning rate for the given iteration
        :param itr:
        :return:
        '''
        return self.lr


class ConstantLrSched(LrSched):
    def __init__(self,initial_learning_rate=0.0001):
        LrSched.__init__(self,name='Constant_lr_Schedule',initial_learning_rate=initial_learning_rate)

    def get_lr(self,itr):
        return self.lr

class Grad_Reduce_LrSched(LrSched):

    def __init__(self,red_period=1,red_coef=0.5,initial_learning_rate=1e-3):
        LrSched.__init__(self,'Diminishing_lr_Schedule')
        self.red_period=red_period
        self.red_coef=red_coef
        self.lr=initial_learning_rate

    def get_lr(self,itr):
        if itr is not 0 and itr%self.red_period is 0:
           self.lr*=self.red_coef
        return self.lr

