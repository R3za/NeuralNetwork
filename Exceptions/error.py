
class LayerInputError(Exception):
    def __init__(self,layer):
        Exception.__init__(self,'Error: the input to the layer {0} is None'.format(layer))



class InputTypeError(Exception):
    def __init__(self,op,type):
        Exception.__init__(self,'datasets type to the {0} should be numpy {1}'.format(op,type))
        # print self.messge
class ParamTypeError(Exception):
    def __init__(self, op_name,param_name ,type_str):
        Exception.__init__(self, 'The input param \'{0}\' to '
                                 'function \'{1}\' should be of type \'{}\''.format(param_name,op_name,type_str))


class KeyNotExistError(Exception):
    def __init__(self,lsit_name,key_str):
        Exception.__init__(self,'There is no entery for \'{0}\' in \'{1}\''.format(key_str,lsit_name))

class NoneError(Exception):
    def __init__(self,param_name,place):
        Exception.__init__(self,'The {0} in the {1} should not be None'.format(param_name,place))
