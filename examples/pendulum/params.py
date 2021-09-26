class PendulumParams(object):
    def __init__(self, dyn_params=None, dyn_bias=None,
                 xuref=None, L_hess=None, L_grad=None, L_const=None,
                 xfref=None, Vf_hess=None, Vf_grad=None, Vf_const=None):
        self.dyn_params = dyn_params
        self.dyn_bias = dyn_bias
        self.xuref = xuref
        self.L_hess = L_hess
        self.L_grad = L_grad
        self.L_const = L_const
        self.xfref = xfref
        self.Vf_hess = Vf_hess
        self.Vf_grad = Vf_grad
        self.Vf_const = Vf_const
