class Regularizer:
    def __init__(self,name="Regularizer"):
        self.name=name

    def get_gradient(self,**kwargs):
        return {}