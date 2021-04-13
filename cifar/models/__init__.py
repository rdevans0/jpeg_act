
from . import vgg
from . import resnet
from . import resnet18
from . import resnet_fast
from . import wide_resnet

models = {
    'vgg':    vgg.VGG,

    'rn8': resnet_fast.ResNet8Fast,
    'rn18': resnet18.ResNet18,
    'rn34': resnet18.ResNet34,
    'rn50': resnet.ResNet50,
    
    'rn101': resnet.ResNet101,
    'rn152': resnet.ResNet152,
    
    'wrn': wide_resnet.WideResNet,
}

