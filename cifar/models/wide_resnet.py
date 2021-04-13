import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers

import sys
import os
import numpy as np

from common.utils import backward_var_iter


def named(code, var, prefix=None, **kwargs):
    if prefix:
        var.name = prefix + code
    else:
        var.name = code
    return var

_default_init = {
    'initialW': initializers.HeNormal(),
#    'initial_gamma': initializers.Constant(1),
#    'initial_beta': initializers.Constant(0),
}

class WideBasic(chainer.Chain):
    
    def __init__(self, n_input, n_output, stride, dropout, init=_default_init):
        super(WideBasic, self).__init__()
        
        self.name = '<WideBasic>'
        self.dropout = dropout
        
        conv_init = dict((k,v) for k,v in init.items() if k in ('initialW','initial_bias'))
        bn_init = dict((k,v) for k,v in init.items() if k in ('initial_gamma','initial_beta'))
        
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                n_input, n_output, 3, stride, 1, nobias=True, **conv_init)
            self.bn1 = L.BatchNormalization(n_input, **bn_init)
            
            self.conv2 = L.Convolution2D(
                n_output, n_output, 3, 1, 1, nobias=True, **conv_init)
            self.bn2 = L.BatchNormalization(n_output, **bn_init)
            
            if n_input != n_output:
                self.shortcut = L.Convolution2D(
                    n_input, n_output, 1, stride, nobias=True, **conv_init)

    def __call__(self, x):
        name = self.name
        
        y = named(name + '_1-n', self.bn1(x))
        y = named(name + '_1-r', F.relu(y))
        
        h = named(name + '_1-c', self.conv1(y))
        h = named(name + '_2-n', self.bn2(h))
        
        xr = '_2-x' if self.dropout else '_2-r'  # Flag as compressible
        h = named(name + xr, F.relu(h))
        
        if self.dropout:
            h =  named(name + '_2-d', F.dropout(h))
        
        h = named(name + '_2-c', self.conv2(h))
        
        if hasattr(self, 'shortcut'):
            b = named(name + '_2-b', self.shortcut(y))
        else:
            b = y
        h = named(name + '_2-s', h + b)
        return h


class WideBlock(chainer.ChainList):
    def __init__(self, name, n_input, n_output, count, stride, dropout):
        super(WideBlock, self).__init__()
        self.name = name
        
        self.basic_blocks = []
        with self.init_scope():
            for i in range(0, count):
                s = stride if i == 0 else 1
                ni = n_input if i == 0 else n_output
                no = n_output
                name = self.name + '_' + str(i)
                
                link = WideBasic(ni, no, s, dropout)
                self.add_link(link)
                link.name = name
                
                self.basic_blocks.append(link)

    def __call__(self, x):
        for i, link in enumerate(self):
            x = link(x)
        return x

class WideResNet(chainer.Chain):
    def __init__(
            self, class_labels=10, widen_factor=10, depth=28,  dropout=True):
        k = widen_factor
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        n_stages = [16, 16 * k, 32 * k, 64 * k]
        w = chainer.initializers.HeNormal()
        
        super(WideResNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                None, n_stages[0], 3, 1, 1, nobias=True, initialW=w)
            self.wide2 = WideBlock('wide2', n_stages[0], n_stages[1], n, 1, dropout)
            self.wide3 = WideBlock('wide3', n_stages[1], n_stages[2], n, 2, dropout)
            self.wide4 = WideBlock('wide4', n_stages[2], n_stages[3], n, 2, dropout)
            self.bn5 = L.BatchNormalization(n_stages[3])
            self.fc6 = L.Linear(n_stages[3], class_labels, initialW=w)
        
        self.layers = [self.wide2, self.wide3, self.wide4]
        
    def __call__(self, x):
        if not isinstance(x, chainer.Variable):
            x = chainer.Variable(x)
        x = named('block1_1-i', x)
        
        h = named('block1_1-c', self.conv1(x))
        h = self.wide2(h)
        h = self.wide3(h)
        h = self.wide4(h)
        
        h = named('final1-n', self.bn5(h))
        h = named('final1-r', F.relu(h))
        h = named('final1-p', F.average_pooling_2d(h, (h.shape[2], h.shape[3])))
        h = named('final2-f', self.fc6(h))
        return h
    
    @property
    def act_names(self):
        if hasattr(self, '_activations_cached'):
            return self._activations_cached
        
        x = self.xp.random.randn(1, 3, 32, 32).astype('f')
        loss = self(x)
        variables = [loss] + [ v for _,_,v in backward_var_iter(loss)]
        
        # Remove duplicates from bottom
        a = [v.name for v in variables if v.data is not None]
        nodup = sorted(list(set(a)), key=list(reversed(a)).index, reverse=1)
        self._activations_cached = nodup
        return self._activations_cached
    
    def act_shapes(self):
        x = self.xp.random.randn(1, 3, 32, 32).astype('f')
        loss = self(x)
        variables = [loss] + [ v for _,_,v in backward_var_iter(loss)]
        
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
    
    
    def namedvars(self):
        return self.custom_namedvars(['block','children','final'])
    
    @property
    def var_names(self):
        """ Covers more activations than act_names """
        predictor = self
        predictor.retain(True)
        x = self.xp.random.randn(16,3,32,32).astype('f')
        #t = self.xp.random.randint(0,1, size=(16,))
        #loss = self(x, t)
        loss = self(x)
        names = [name for name,_ in predictor.namedvars()]
        del loss
        predictor.retain(False)
        return names


if __name__ == '__main__':
    
    model = WideResNet(10)
    
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
        
            
    print('Activation Size %8.2f MB (bs=1)'%(act_numel * 4. / 1024 / 1024))
    print('Parameter Size %8.2f MB (10 classes)'%(param_numel * 4. / 1024 / 1024))
