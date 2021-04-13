import chainer
import chainer.links as L
import chainer.functions as F

from chainer import initializers
from functools import partial

from common.utils import backward_var_iter

def _set_name(code, var, prefix=None):
    if prefix:
        var.name = prefix + code
    else:
        var.name = code
    return var


class BottleNeckA(chainer.Chain):
    """ Downsampling 2 layer bottleneck, weighted bypass """

    def __init__(self, in_size, ch, out_size, stride=2):
        super(BottleNeckA, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_size, ch, 3, stride, 1, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, out_size, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(out_size)

            self.conv3 = L.Convolution2D(
                in_size, out_size, 1, stride, 0,
                initialW=initialW, nobias=True)
            self.bn3 = L.BatchNormalization(out_size)

    def __call__(self, x):
        named = partial(_set_name, prefix=self.name)
        h = named('_1-c', self.conv1(x))
        h = named('_1-n', self.bn1(h))
        h = named('_1-r', F.relu(h))
        
        h = named('_2-c', self.conv2(h))
        h = named('_2-n', self.bn2(h))
        
        b = named('_3-c', self.conv3(x))
        b = named('_3-n', self.bn3(b))
        
        s = named('_3-s', h + b)
        s = named('_3-r', F.relu(s))

        return s


class BottleNeckB(chainer.Chain):
    """ 2 layer bottleneck, unit bypass """

    def __init__(self, ch):
        super(BottleNeckB, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(
                ch, ch, 3, 1, 1, initialW=initialW, nobias=True)
            self.bn2 = L.BatchNormalization(ch)
            
    def __call__(self, x):
        named = partial(_set_name, prefix=self.name)
        h = named('_1-c', self.conv1(x))
        h = named('_1-n', self.bn1(h))
        h = named('_1-r', F.relu(h))
        
        h = named('_2-c', self.conv2(h))
        h = named('_2-n', self.bn2(h))

        s = named('_2-s', h + x)
        s = named('_2-r', F.relu(s))
        
        return s


class Block(chainer.ChainList):

    def __init__(self, name, layer, in_size, ch, out_size, stride=2):
        super(Block, self).__init__()
        self.name = name
        link = BottleNeckA(in_size, ch, out_size, stride)
        self.add_link(link)
        link.name = '{}_{}'.format(self.name, link.name)
        for i in range(layer - 1):
            link = BottleNeckB(ch)
            self.add_link(link)
            link.name = '{}_{}'.format(self.name, link.name)

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x
    
class ResNetAbstract(chainer.Chain):
    def __init__(self):
        super(ResNetAbstract, self).__init__()
        
    @property
    def act_names(self):
        if hasattr(self, '_activations_cached') and self._activations_cached is not None:
            return self._activations_cached 
        x = self.xp.zeros((4,3,32,32)).astype('f')
        
        loss = self(x)
        variables = [loss] + [ v for _,_,v in backward_var_iter(loss)]
        self._activations_cached = [v.name for v in variables if v.data is not None]
        return self._activations_cached 
    
    def post_scramble_callback(self, loss, scrambler_map):
        pass

class ResNet18(ResNetAbstract):
    insize = 32
    def __init__(self, class_labels=10):
        super(ResNet18, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 3, 1, 1, True, 
                                         initialW=initializers.HeNormal())
            self.bn1 = L.BatchNormalization(64)
            self.res2 = Block('res2', 2,   64,  64,  64, stride=1)
            self.res3 = Block('res3', 2,   64, 128, 128)
            self.res4 = Block('res4', 2,  128, 256, 256)
            self.res5 = Block('res5', 2,  256, 512, 512)
            self.fc = L.Linear(512, class_labels)

    def __call__(self, x):
        named = partial(_set_name, prefix='block1_1')
        h = named('-c', self.conv1(x))
        h = named('-n', self.bn1(h))
        h = named('-r', F.relu(h))
        h = named('-p', F.max_pooling_2d(h, 3, stride=2))
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        
        named = partial(_set_name, prefix='final1')
        h = named('-p',F.average_pooling_2d(h, h.shape[2:]))
        h = named('-f', self.fc(h))
        return h
    
class ResNet34(ResNetAbstract):
    insize = 32
    def __init__(self, class_labels=10):
        super(ResNet34, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, 64, 7, 2, 3, initialW=initializers.HeNormal())
            self.bn1 = L.BatchNormalization(64)
            self.res2 = Block('res2', 3,   64,  64,  64, stride=1)
            self.res3 = Block('res3', 4,   64, 128, 128)
            self.res4 = Block('res4', 6,  128, 256, 256)
            self.res5 = Block('res5', 3,  256, 512, 512)
            self.fc = L.Linear(512, class_labels)

    def __call__(self, x):
        named = partial(_set_name, prefix='block1_1')
        h = named('-c', self.conv1(x))
        h = named('-n', self.bn1(h))
        h = named('-r', F.relu(h))
        h = named('-p', F.max_pooling_2d(h, 3, stride=2))
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        
        named = partial(_set_name, prefix='final1')
        h = named('-p', F.average_pooling_2d(h, h.shape[2:]))
        h = named('-f', self.fc(h))
        return h
    