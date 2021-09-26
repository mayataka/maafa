class CartpoleParams(object):
    def __init__(self, dyn_params=None, 
                 xuref=None, L_hess=None, 
                 xfref=None, Vf_hess=None):
        self.dyn_params = dyn_params
        self.xuref = xuref
        self.L_hess = L_hess
        self.xfref = xfref
        self.Vf_hess = Vf_hess