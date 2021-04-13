
from functools import partial

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import backward_var_iter_nodup

def add(a,b):
    return a+b

def named(name, var):
    var.name = name
    return var

class ConvBN(chainer.Chain):
    def __init__(self, name, c_in, c_out, pool=None):
        super(ConvBN, self).__init__()
        self.name = name
        w = chainer.initializers.HeNormal()
        #1/np.sqrt(6)
        
        with self.init_scope():
            self.conv = L.Convolution2D(c_in, c_out, ksize=3, stride=1, pad=1, 
                                        nobias=True, initialW=w)
            self.bn = L.BatchNormalization(c_out)
            if pool is not None:
                self.pool = pool
            
        self.relu = F.relu
    
    def __call__(self, x):
        h = named(self.name + '-c', self.conv(x))
        h = named(self.name + '-n', self.bn(h))
        h = named(self.name + '-r', self.relu(h))
        if hasattr(self, 'pool'):
            h = named(self.name + '-p', self.pool(h))
        
        return h

class Residual(chainer.Chain):
    def __init__(self, name, c_in, c_out, pool=None, **kw):
        super(Residual, self).__init__()
        self.name = name
        
        with self.init_scope():
            self.conv_bn1 = ConvBN('', c_in, c_out, pool=pool, **kw)
            self.conv_bn2 = ConvBN('', c_out, c_out, pool=None, **kw)
            self.conv_bn3 = ConvBN('', c_out, c_out, pool=None, **kw)
        self.add = add
        self.conv_bn1.name = self.name + '_1'
        self.conv_bn2.name = self.name + '_2'
        self.conv_bn3.name = self.name + '_3'
    
    def __call__(self, x):
        y = self.conv_bn1(x)
        h = self.conv_bn2(y)
        h = self.conv_bn3(h)
        s = named(self.name + '_3-s', self.add(h, y))
        return s

class ResNet8Fast(chainer.Chain):
    def __init__(self, n_class=10, layers=None, weight=0.125, **kw):
        super(ResNet8Fast, self).__init__()
        
        layers = layers or [('block1_1', 64), ('block2_1',128), ('res3_0',256), ('res4_0',512)]
        channels = [c for _,c in layers]
        
        self.layers = []
        
        with self.init_scope():
            for layer_ind, (layer_name, c_out) in enumerate(layers):
                if layer_ind == 0:
                    c_in = 3
                    pool = None
                else:
                    c_in = channels[layer_ind-1]
                    pool = partial(F.max_pooling_2d, ksize=2, stride=2)
                
                if 'res' in layer_name:
                    layer = Residual(layer_name, c_in, c_out, pool=pool, **kw)
                else:
                    layer = ConvBN(layer_name, c_in, c_out, pool=pool, **kw)
                    
                self.layers.append(layer)
                setattr(self, layer_name, layer)
                
            self.linear = L.Linear(channels[-1], 10, nobias=True)
        self.pool = partial(F.max_pooling_2d, ksize=4, stride=4)
        self.weight = weight
        
    def __call__(self, x):
        if not isinstance(x, chainer.Variable):
            x = chainer.Variable(x)
        h = x
        h.name = 'input'
        for layer in self.layers:
            h = layer(h)
        
        h = named('final1-p', self.pool(h))
        h = named('final1-f', self.linear(h))
        h = h*self.weight
        return h
    
    
    @property
    def act_names(self):
        if hasattr(self, '_activations_cached'):
            return self._activations_cached
        
        x = self.xp.random.randn(1, 3, 32, 32).astype('f')
        loss = self(x)
        variables = [loss] + [ v for _,_,v in backward_var_iter_nodup(loss)]
        
        # Remove duplicates from bottom
        a = [v.name for v in variables if v.data is not None]
        nodup = sorted(list(set(a)), key=list(reversed(a)).index, reverse=1)
        self._activations_cached = nodup
        return self._activations_cached
    
    def act_shapes(self):
        x = self.xp.random.randn(1, 3, 32, 32).astype('f')
        loss = self(x)
        variables = [loss] + [ v for _,_,v in backward_var_iter_nodup(loss)]
        
        # Remove duplicates from bottom
        a = [v.name for v in variables if v.data is not None]
        shape_map = dict((v.name,v.shape) for v in variables if v.data is not None)
        nodup = sorted(list(set(a)), key=list(reversed(a)).index, reverse=1)
        total = 0
        shapes = []
        for name in nodup:
            shape = shape_map[name]
            shapes += [(name, shape)]
            total += np.prod(shape)
        
        return shapes, total
            
    def post_scramble_callback(self, loss, scrambler_map):
        pass

if __name__ == '__main__':
    
    import sys
    for model_class in (ResNet8Fast,):
        model = model_class(10)
        
        act_shapes, act_numel = model.act_shapes()
        param_numel = sum(np.prod(p.shape) for p in model.params())
        
        print(repr(model.__class__))
        
        if '-v' in sys.argv:
            print("'{}': {{".format(model.__class__.__name__))
            print(" 'weights': [")
            for name, param in model.namedparams():
                print("  ('{}', {}),".format(name, param.shape))
            print("],")
            print(" 'activations': [")
            for name, shape in reversed(act_shapes):
                print("  ('{}', {}),".format(name, shape))
            print("]},")
        
        suffs = ['c','r','s','d']
        if '-b' in sys.argv:
            print('Breakdown by activation type:')
            counted = 0
            for suff in suffs:
                suff_numel = sum([np.prod(shape) for name,shape in act_shapes if name and '-'+suff in name])
                counted += suff_numel
                print('  "-{}": {:8.2f}, #MB (bs=1) {:.1f}%'.format(suff, suff_numel * 4./1024/1024, 100.*suff_numel/act_numel))
            rem_numel = act_numel - counted
            print('  "other": {:8.2f}, #MB (bs=1) {:.1f}%'.format(rem_numel * 4./1024/1024, 100.*rem_numel/act_numel))
            
            
                
        print('Activation Size %8.2f MB (bs=1)'%(act_numel * 4. / 1024 / 1024))
        print('Parameter Size %8.2f MB (10 classes)'%(param_numel * 4. / 1024 / 1024))
