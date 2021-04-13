

import chainer
import numpy as np

class Helper:
    def scramble_setup(self):
        pass
        
    def print_settings(self, printfn=print):
        pass
       
    def scramble(self,rank, function, var, force_signed=False):
        raise NotImplementedError
       
    def ascramble(self, array, force_signed=False):
        """ Scrambles a numpy or cupy array, in place """
        var = chainer.Variable(array)
        self.scramble(None, None, var, force_signed=force_signed)
        return True
    
    def get_sign_cached(self, x, rank, func, var):
        if self.cache is not None:
            key = (rank, var.name, func.label if func is not None else None)
            if key in self.cache:
                signed = self.cache[key]
            else:
                signed = int(x.min() <= 0)
                self.cache[key] = signed
        else:
            signed = int(x.min() <= 0)
        return signed
    
    def setup_cache(self):
        if not hasattr(self, 'cache') or self.cache is None:
            self.cache = {} # Starting the cache


class DestroyHelper(Helper):
    def __init__(self):
        self.name = 'destroy'
        self.detail = 'destroy'
        
    def scramble(self, r, f, var, force_signed=False):
        ''' Kills the activations '''
        var.data *= np.Inf

