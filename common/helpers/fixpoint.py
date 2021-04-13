
from chainer import cuda
import argparse

import sys, os
sys.path.append(os.path.abspath('..'))

from common.helpers import Helper


class FixPointHelper(Helper):
    def __init__(self, whole_bits, frac_bits=None, cache_sign=True):
        if frac_bits is None:
            bits = whole_bits
        else:
            bits = (int(whole_bits), int(frac_bits))
        if len(bits) is not 2 or type(bits[0]) != int or type(bits[1]) != int:
            argparse.ArgumentTypeError('fix point has invalid specification {}'.format(bits))
            
        self.bits = bits
        self.cache_sign = cache_sign
        self.name = 'fixpoint'
        self.detail = 'fixpoint-{}.{}'.format(self.bits[0], bits[1])
    
    def scramble_setup(self):
        if self.cache_sign:
            self.setup_cache()
        
    def print_settings(self, printfn=print):
        printfn('#   bits: {}'.format(self.bits))
        printfn('#   shift: x{}'.format(2.0**self.bits[1]))
        
    def scramble(self,rank, function, var, force_signed=False):
        ''' Introduce error to the variable '''
        
        if var.data is None:
            raise Exception('None encountered in {}'.format(var.name))
        
        x  = var.data
        xp = cuda.get_array_module(x)
        #num_imgs, num_chan, w,h = x.shape
        
        signed = force_signed or self.get_sign_cached(x, rank, function, var)
        
        # Calculate signed transform to an fixpoint int
        wbits,fbits = self.bits   # Whole and fractional bits
        b = wbits + fbits     # Total bits
        
        # We add a bias for signed values
        # The range becomes 
        #  upper bound    2^b - 2^(b-1) - 1
        #  lower bound   -2^b + 2^(b-1)
        upper = 2**b - 2**(b-1) - 1
        lower = -2**b + 2**(b-1)
        shift = 2.0**fbits
        bias = lower if not signed else 0
        
        # Equivalent to below but with in-place operations
        # x = (x*shifts + bias).clip(upper, lower).trunc() 
        x *= shift
        if bias != 0:
            x += bias
        xp.clip(x, lower, upper, out=x)
        #xp.trunc(x, out=x)
        xp.rint(x, out=x)
        
        
        # Undo shifting
        if bias != 0:
            x -= bias
        x *= (1./shift)
        
        
        return True
    
class Fix35Helper(Helper):
    """ Hard coded fixpoint helper. Should be significantly faster """
    
    def __init__(self, cache_sign=True):
        self.name = 'fix35'
        self.detail = 'fix35'
        self.cache_sign = cache_sign
        self.cache = None
    
    def scramble_setup(self):
        if self.cache_sign and self.cache is None:
            self.cache = {} # Starting the cache
        
    def print_settings(self, printfn=print):
        printfn('#   bits: {}'.format((3,5)))
        printfn('#   shift: x{}'.format(2.0**5))
        
    def scramble(self,rank, function, var, force_signed=False):
        ''' Introduce error to the variable '''
        
        if var.data is None:
            raise Exception('None encountered in {}'.format(var.name))
        
        x  = var.data
        if self.cache is not None:
            key = (rank, var.name, function.label if function is not None else None)
            if key in self.cache:
                signed = self.cache[key]
            else:
                signed = int(force_signed or x.min() < 0)
                self.cache[key] = signed
        else:
            signed = int(force_signed or x.min() < 0)
        bias = 0 if signed else -128
        cuda.elementwise(
            'float32 bias',
            'float32 x',
            ''' 
                float shifted = ldexp(x, 5) + bias;           // shifted = x * 32 + bias
                
                int iScram = __float2int_rn(shifted);
                float scram = __int2float_rn( (iScram<-128) ? -128 : 
                                              (iScram >=127) ? 127 : iScram );  // Clip and 2float
                
                x = ldexp(scram - bias, -5);
            ''', 
            'scramble_fix35') (bias, x)
        
        return True

    
    

if __name__ == '__main__':
    # Tests the performance of standard and cupy implementation
    import numpy as np
    import time
    dev = 0
    for _ in range(10):
        helpers = {}
        helpers['FixHelper'] = FixPointHelper(3,5)
        helpers['Fix35'] = Fix35Helper()
        output = {}
        data = cuda.to_gpu(8*np.random.randn(64,32,32,32).astype('f'), device=dev)
        if np.random.rand() > 0.5:
            data = abs(data)
        for name, h in helpers.items():
            h.scramble_setup()
            x = data.copy()
            st = time.time()
            h.ascramble(x)
            et = time.time()
            output[name] = x.copy()
            print('{:10}\t{:10} seconds'.format(name, et - st))
        for name, o in output.items():
            if name is 'FixHelper':
                continue
            diff = o - output['FixHelper']
            print('{} diff to FixHelper: {}'.format(name, abs(diff).max()))
        print('')
        
    print(data[0,0,0,:5])
    print(output['FixHelper'][0,0,0,:5])
    print(output['Fix35'][0,0,0,:5])
