from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L

import numpy as np

from common.utils import backward_var_iter_nodup


def named(code, var, prefix=None, **kwargs):
    if prefix:
        var.name = prefix + code
    else:
        var.name = code
    return var

class Block(chainer.Chain):

    """A convolution, batch norm, ReLU block.
    
    Named activations for convenience

    A block in a feedforward network that performs a
    convolution followed by batch normalization followed
    by a ReLU activation.

    For the convolution operation, a square filter size is used.

    Args:
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter is ksize x ksize.
        pad (int): The padding to use for the convolution.

    """

    def __init__(self, out_channels, ksize, name, pad=1, **kwargs):
        super(Block, self).__init__()
        
        self.relu =  kwargs['relu_class'] if 'relu_class' in kwargs else F.relu
                
        with self.init_scope():
            self.name = name
            self.conv = L.Convolution2D(None, out_channels, ksize, pad=pad,
                                        nobias=True)
            if 'batchnorm_class' in kwargs:
                self.bn = kwargs['batchnorm_class'](out_channels)
            else:
                self.bn = L.BatchNormalization(out_channels)
            
            self.layers = [self.name + suffix for suffix in ('-c','-n','-r')]

    def __call__(self, x):
        name = self.name
        
        h = named(name + '-c', self.conv(x))
        h = named(name + '-n', self.bn(h))
        h = named(name + '-x', self.relu(h))
        return h
    
    def child_iter(self):
        """ Override to have no children, layers is used for something else"""
        return
        yield
    
class BlockCR(chainer.Chain):

    """A Block that with Conv and ReLU only

    Args:
        out_channels (int): The number of output channels.
        ksize (int): The size of the filter is ksize x ksize.
        pad (int): The padding to use for the convolution.

    """

    def __init__(self, out_channels, ksize, name, pad=1):
        super(BlockCR, self).__init__()
        with self.init_scope():
            self.name = name
            self.conv = L.Convolution2D(None, out_channels, ksize, pad=pad,
                                        nobias=True)
            
            self.layers = [self.name + suffix for suffix in ('-c','-r')]

    def __call__(self, x):
        h = self.conv(x)
        h.name = self.name + '-c'
        h = F.relu(h)
        h.name = self.name + '-r'
        return h

class Abstract_VGG(chainer.Chain):

    """ Requires feature detector and classifier to be initialized in __init__ """
    
    LAYERS = [
        'final2-f',   'final2-d',
        'final1-r',   'final1-n',   'final1-f',   None,         'final1-d',
        'block5_3-p', 'block5_3-r', 'block5_3-n', 'block5_3-c',
        'block5_2-d', 'block5_2-r', 'block5_2-n', 'block5_2-c',
        'block5_1-d', 'block5_1-r', 'block5_1-n', 'block5_1-c',
        'block4_3-p', 'block4_3-r', 'block4_3-n', 'block4_3-c',
        'block4_2-d', 'block4_2-r', 'block4_2-n', 'block4_2-c',
        'block4_1-d', 'block4_1-r', 'block4_1-n', 'block4_1-c',
        'block3_3-p', 'block3_3-r', 'block3_3-n', 'block3_3-c',
        'block3_2-d', 'block3_2-r', 'block3_2-n', 'block3_2-c',
        'block3_1-d', 'block3_1-r', 'block3_1-n', 'block3_1-c',
        'block2_2-p', 'block2_2-r', 'block2_2-n', 'block2_2-c',
        'block2_1-d', 'block2_1-r', 'block2_1-n', 'block2_1-c',
        'block1_2-p', 'block1_2-r', 'block1_2-n', 'block1_2-c',
        'block1_1-d', 'block1_1-r', 'block1_1-n', 'block1_1-c',
    ]

    def __init__(self):
        super(Abstract_VGG, self).__init__()
    
    def __call__(self, x):
        h = self.feature_detector(x)
        h = self.classifier(h)
        return h
    
    @property
    def act_names(self):
        if hasattr(self, '_activations_cached') and self._activations_cached is not None:
            return self._activations_cached 
        x = self.xp.zeros((64,3,32,32)).astype('f')
        
        loss = self(x)
        variables = [loss] + [ v for _,_,v in backward_var_iter_nodup(loss)]
        self._activations_cached = [v.name for v in variables if v.data is not None]
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
    
    def init_feature_detector(self, block=Block, upto3_3=None, block1_1=None, channel_ratio=64, **kwargs):
        """ Common function for feature detector initialization
        
        All arguments are chainer.Chain and must have the following argument list:
            out_channels, ksize, name, pad=1, (optional) **kwargs
    
        Args:
            self (chainer.Chain): self keyword from calling class
            block (chainer.Chain): Default block
            upto3_3 (chainer.Chain): Overrides block for layers 1_1 until 3_2 (inclusive)
            block1_1 (chainer.Chain): Overrides block and upto3_3 for block1_1
            
        """
        upto3_3 = upto3_3 if upto3_3 is not None else block
        block1_1 = block1_1 if block1_1 is not None else block
        cr = channel_ratio
        with self.init_scope():
            self.block1_1 = block1_1(1*cr, 3,  '1_1', **kwargs)
            self.block1_2 = upto3_3(1*cr, 3,  '1_2', **kwargs)
            self.block2_1 = upto3_3(2*cr, 3, '2_1', **kwargs)
            self.block2_2 = upto3_3(2*cr, 3, '2_2', **kwargs)
            self.block3_1 = upto3_3(4*cr, 3, '3_1', **kwargs)
            self.block3_2 = upto3_3(4*cr, 3, '3_2', **kwargs)
            self.block3_3 = upto3_3(4*cr, 3, '3_3', **kwargs)
            self.block4_1 = block(8*cr, 3, '4_1', **kwargs)
            self.block4_2 = block(8*cr, 3, '4_2', **kwargs)
            self.block4_3 = block(8*cr, 3, '4_3', **kwargs)
            self.block5_1 = block(8*cr, 3, '5_1', **kwargs)
            self.block5_2 = block(8*cr, 3, '5_2', **kwargs)
            self.block5_3 = block(8*cr, 3, '5_3', **kwargs)
        
        self.layers = [self.block1_1, self.block1_2, 
                       self.block2_1, self.block2_2,
                       self.block3_1, self.block3_2, self.block3_3,
                       self.block4_1, self.block4_2, self.block4_3,
                       self.block5_1, self.block5_2, self.block5_3]
    
    def init_classifier(self, class_labels=10, num_hidden=512):
        """ Common function for classifier initialization
    
        Args:
            class_labels (int): number of output neurons
            num_hidden (int): number of hidden neurons
        """
        
        with self.init_scope():
            self.fc1 = L.Linear(None, num_hidden, nobias=True)
            self.bn_fc1 = L.BatchNormalization(num_hidden)
            self.fc2 = L.Linear(None, class_labels, nobias=True)
            
    
    def feature_detector(self, h, dropout_ratio=(0.3, 0.4)):
        dr = dropout_ratio
        
        # 64 channel blocks:
        h = self.block1_1(h)
        h = named('block1_1-d', F.dropout(h, ratio=dr[0]))
        h = self.block1_2(h)
        with chainer.using_config('use_cudnn','never'):
            h = named('block1_2-p', F.max_pooling_2d(h, ksize=2, stride=2))
    
        # 128 channel blocks:
        h = self.block2_1(h)
        h = named('block2_1-d', F.dropout(h, ratio=dr[1]))
        h = self.block2_2(h)
        with chainer.using_config('use_cudnn','never'):
            h = named('block2_2-p', F.max_pooling_2d(h, ksize=2, stride=2))
    
        # 256 channel blocks:
        h = self.block3_1(h)
        h = named('block3_1-d', F.dropout(h, ratio=dr[1]))
        h = self.block3_2(h)
        h = named('block3_2-d', F.dropout(h, ratio=dr[1]))
        h = self.block3_3(h)
        with chainer.using_config('use_cudnn','never'):
            h = named('block3_3-p', F.max_pooling_2d(h, ksize=2, stride=2))
        
        # 512 channel blocks:
        h = self.block4_1(h)
        h = named('block4_1-d', F.dropout(h, ratio=dr[1]))
        h = self.block4_2(h)
        h = named('block4_2-d', F.dropout(h, ratio=dr[1]))
        h = self.block4_3(h)
        with chainer.using_config('use_cudnn','never'):
            h = named('block4_3-p', F.max_pooling_2d(h, ksize=2, stride=2))
    
        # 512 channel blocks:
        h = self.block5_1(h)
        h = named('block5_1-d', F.dropout(h, ratio=dr[1]))
        h = self.block5_2(h)
        h = named('block5_2-d', F.dropout(h, ratio=dr[1]))
        h = self.block5_3(h)
        with chainer.using_config('use_cudnn','never'):
            h = named('block5_3-p', F.max_pooling_2d(h, ksize=2, stride=2))
        return h
    
    def classifier(self, h, dropout_ratio=0.5):
        h = named('final1-d', F.dropout(h, ratio=dropout_ratio))
        h = named('final1-f', self.fc1(h))
        h = named('final1-n', self.bn_fc1(h))
        h = named('final1-r', F.relu(h))
        h = named('final2-d', F.dropout(h, ratio=dropout_ratio))
        h = named('final2-f', self.fc2(h))
        return h



class VGG(Abstract_VGG):
    def __init__(self, class_labels=10):
        super(VGG, self).__init__()
        self.init_feature_detector(block=Block)
        self.init_classifier(class_labels=class_labels)
    
    
    
class VGG_flex(Abstract_VGG):

    """ Much more flexible VGG
    
    Args:
        class_labels (int): The number of class labels.
        
        args (defaultdict(dict)): arguments to tune the network configuration
        
    
        VarTracking.__init__(self)
    Available Extra Args (for ``arg``):
        dropout/ratio (float or tuple of floats): default = (0.3, 0.4, 0.5)
        
        block/batchnorm_class (Link): Class to use for block batch normalization
    
    """

    def __init__(self, class_labels=10, kwargs={}):
        super(VGG_flex, self).__init__()
        
        # Flex Argument Checking 
        def default_arg(key, default, allowed_types=None):
            if key not in kwargs:
                kwargs[key] = default
            if allowed_types is not None:
                if type(kwargs[key]) not in allowed_types:
                    raise ValueError(
                                     'Argument {}={} is not in allowed types {}'.format(
                                     key, kwargs[key], allowed_types))
        
        default_arg('dropout/ratio', (0.3, 0.4, 0.5), (int, float, tuple, list))
        
        if type(kwargs['dropout/ratio']) in (int, float):
            kwargs['dropout/ratio'] = (kwargs['dropout/ratio'],)*3
        
        
        self.kwargs = kwargs
        block_kwargs = dict((k.replace('block/',''),v) for k,v in kwargs.items() if k.startswith('block/'))
        
        # Flex init
        self.init_feature_detector(block=Block, **block_kwargs)
        self.init_classifier(class_labels)

    def __call__(self, x):
        h = self.feature_detector(x, 
                dropout_ratio=self.kwargs['dropout/ratio'][:2])
        h = self.classifier(h, 
                dropout_ratio=self.kwargs['dropout/ratio'][2])
        return h


                
            
class VGG_small(Abstract_VGG):
    """ Channels are 32, 32, 64 ... etc. instead of 64,64,128... """
    def __init__(self, class_labels=10, kwargs={}):
        super(VGG_small, self).__init__()
        self.init_feature_detector(block=Block, channel_ratio=32)
        self.init_classifier(class_labels=class_labels)
        

class VGG_big(Abstract_VGG):
    """ Classifier has 1024 hidden neurons and includes and extra 1024->1024 layer"""
    def __init__(self, class_labels=10, kwargs={}):
        super(VGG_big, self).__init__()
        self.init_feature_detector(block=Block)
        self.init_classifier(class_labels=class_labels)
        
    def init_classifier(self, class_labels=10):
        with self.init_scope():
            self.fc1 = L.Linear(None, 1024, nobias=True)
            self.fc2 = L.Linear(1024, 1024, nobias=True)
            self.fc3 = L.Linear(None, class_labels, nobias=True)
        
    def classifier(self, h):
        h = F.dropout(h, ratio=0.5)
        h.name = 'final1-d'
        h = self.fc1(h)
        h.name = 'final1-f'
        h = F.relu(h)
        h.name = 'final1-r'
        
        h = F.dropout(h, ratio=0.5)
        h.name = 'final2-d'
        h = self.fc2(h)
        h.name = 'final2-f'
        h = F.relu(h)
        h.name = 'final2-r'
        
        h = self.fc3(h)
        h.name = 'final3-f'
        return h
        

        
            
            
if __name__ == '__main__':
    
    import numpy as np
    import sys
    model = VGG(10)
    
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
