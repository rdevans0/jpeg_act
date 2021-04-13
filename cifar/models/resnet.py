import chainer
import chainer.functions as F
import chainer.links as L
import sys
#import os
import numpy as np

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import backward_var_iter_nodup
#sys.path.pop()

def named(name, var):
    var.name = name
    return var

class BottleNeck(chainer.Chain):

    def __init__(self, n_in, n_mid, n_out, stride=1, use_conv=False, relu=F.relu):
        w = chainer.initializers.HeNormal()
        super(BottleNeck, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(n_in, n_mid, 1, stride, 0, True, w)
            self.bn1 = L.BatchNormalization(n_mid)
            self.conv2 = L.Convolution2D(n_mid, n_mid, 3, 1, 1, True, w)
            self.bn2 = L.BatchNormalization(n_mid)
            self.conv3 = L.Convolution2D(n_mid, n_out, 1, 1, 0, True, w)
            self.bn3 = L.BatchNormalization(n_out)
            if use_conv:
                self.conv4 = L.Convolution2D(
                    n_in, n_out, 1, stride, 0, True, w)
                self.bn4 = L.BatchNormalization(n_out)
        self.use_conv = use_conv
        self.relu = relu

    def __call__(self, x):
        name = self.name
        
        h = named(name + '_1-c', self.conv1(x))
        h = named(name + '_1-n', self.bn1(h))
        h = named(name + '_1-r', self.relu(h))
        
        h = named(name + '_2-c', self.conv2(h))
        h = named(name + '_2-n', self.bn2(h))
        h = named(name + '_2-r', self.relu(h))
        
        
        h = named(name + '_3-c', self.conv3(h))
        h = named(name + '_3-n', self.bn3(h))
        
        if self.use_conv:
            bp = named(name + '_4-c', self.conv4(x))
            bp = named(name + '_4-n', self.bn4(bp))
        else:
            bp = x
        
        h = named(name + '_4-s', h + bp)
        return h


class Block(chainer.ChainList):

    def __init__(self, name, n_in, n_mid, n_out, n_bottlenecks, stride=2, relu=F.relu):
        super(Block, self).__init__()
        self.name = name
        link = BottleNeck(n_in, n_mid, n_out, stride, True, relu=relu)
        self.add_link(link)
        link.name = '{}_{}'.format(self.name, link.name)
        for _ in range(n_bottlenecks - 1):
            link = BottleNeck(n_out, n_mid, n_out, relu=relu)
            self.add_link(link)
            link.name = '{}_{}'.format(self.name, link.name)

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class ResNet(chainer.Chain):

    def __init__(self, n_class=10, n_blocks=[3, 4, 6, 3], relu=F.relu):
        super(ResNet, self).__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 3, 1, 1, True, w)
            self.bn1 = L.BatchNormalization(64)
            self.res2 = Block('res2', 64, 64, 256, n_blocks[0], 1, relu=relu)
            self.res3 = Block('res3',256, 128, 512, n_blocks[1], 2, relu=relu)
            self.res4 = Block('res4',512, 256, 1024, n_blocks[2], 2, relu=relu)
            self.res5 = Block('res5',1024, 512, 2048, n_blocks[3], 2, relu=relu)
            self.fc6 = L.Linear(None, n_class)
        self.relu = relu
        
        self.layers = [self.res2, self.res3, self.res4, self.res5]

    def __call__(self, x):
        
        h = named('block1_1-c', self.conv1(x))
        h = named('block1_1-n', self.bn1(h))
        h = named('block1_1-r', self.relu(h))
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = named('final1-p', F.average_pooling_2d(h, h.shape[2:]))
        h = named('final1-c', self.fc6(h))
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
    
    def namedvars(self):
        return self.custom_namedvars(['block','children','final'])
        

    def post_scramble_callback(self, loss, scrambler_map):
        pass
    

def _var_names(self):
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
    

class ResNet50(ResNet):
    def __init__(self, class_labels=10):
        super(ResNet50, self).__init__(class_labels, [3, 4, 6, 3])


class ResNet101(ResNet):
    def __init__(self, class_labels=10):
        super(ResNet101, self).__init__(class_labels, [3, 4, 23, 3])


class ResNet152(ResNet):
    def __init__(self, class_labels=10):
        super(ResNet152, self).__init__(class_labels, [3, 8, 36, 3])
        

if __name__ == '__main__':
    
    import sys
    for model_class in (ResNet50, ResNet101):
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
