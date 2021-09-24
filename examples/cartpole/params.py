class CartpoleParams(object):
    def __init__(self, dyn_params=None, 
                 xuref=None, xuweight=None, 
                 xfref=None, xfweight=None):
        self.dyn_params = dyn_params
        self.xuref = xuref
        self.xuweight = xuweight
        self.xfref = xfref
        self.xfweight = xfweight