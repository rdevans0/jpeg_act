
from __future__ import print_function
import os
import sys
import six
import argparse
import collections
import copy

from functools import partial

import chainer
import chainer.links as L
import chainer.functions as F

from chainer import cuda
from chainer import training
from chainer.training import extensions
from chainer.training.extensions import log_report as log_report_module

from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100
from chainer.datasets import TransformDataset


from skimage import transform as skimage_transform
from chainercv import transforms
import numpy as np


from models import models


def main(argv=sys.argv[1:]):
    print('CIFAR10/100 (%s)'%__file__)
    if type(argv) == str:
        argv = argv.split()
        
    parser = argparse.ArgumentParser(description='Chainer CIFAR Example')
    
    # Command line arguments
    add_base_args(parser)
    args = parser.parse_args()
    
    # Other settings and derived arguments
    end_trigger = (args.epoch, 'epoch')
    report_file = os.path.join(args.out, 'report.txt')
    report_entries = [
        'epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'lr', 'elapsed_time'
    ]
    
    
    # Header, and output directory
    if not os.path.exists(args.out): 
        os.mkdir(args.out)
    open(report_file,'w').close() # Clears report
    report = open(report_file, 'a')
    print_header(args, argv, log=report, preamble='CIFAR10/100 (%s)'%__file__)
    
    
    ##
    # Set up model and dataset iterators
    rng, fixed_seeds = seed_rng(args.seed, args.gpu)
    train_iter, val_iter, class_labels = load_dataset(args.batchsize, args.dataset, 
                                                      args.augment, args.fast,
                                                      args.old_test_method)
    model = init_model(models[args.model], class_labels=class_labels, gpu=args.gpu, fast=args.fast)

    # Set up an optimizer
    lr, lr_ext, lr_trigger = get_lr_schedule(args, train_iter, fast=args.fast)
    if args.fast:
        optimizer = NesterovAGLossHooks(lr=lr, momentum=args.momentum)
    else:
        optimizer = MomentumSGDLossHooks(lr=lr, momentum=args.momentum)
    optimizer.setup(model)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, end_trigger, out=args.out)
    
    # Decay
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
    trainer.extend(lr_ext, trigger=lr_trigger)
    
    # Extensions - Measurements
    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu))
    trainer.extend(extensions.observe_lr())
    
    # Extensions - Logging
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(report_entries))
    trainer.extend(PrintReportNoSpecial(report_entries, out=report))
    trainer.extend(extensions.ProgressBar(update_interval=args.update_interval))
    
    # Extensions - Snapshots
    trainer.extend(extensions.snapshot(), trigger=end_trigger)
    if args.snapshot_every:
        trainer.extend(extensions.snapshot(
                filename='snapshot_{0.updater.epoch}_iter_{0.updater.iteration}'), 
                trigger=(args.snapshot_every, 'epoch'))
    
    ##
    # Resume Training
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
        
    ##
    # Run the training
    trainer.run()
    
    report.close()

def add_base_args(parser):
    parser.add_argument('--model','-m', default='VGG',
                        help='Model to use for classification (options: *VGGL, VGG)')
    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    
    parser.add_argument('--epoch', '-e', type=float, default=300,
                        help='Number of sweeps over the dataset to train')
    
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    
    parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                        help='Learning rate for SGD')
    
    parser.add_argument('--momentum', '-p', type=float, default=0.9,
                        help='Momentum for SGD')
    
    parser.add_argument('--learnrate_decay', '-y', type=float, default=25,
                        help='Number of epochs to decrease learning rate after (default: 25)')
    
    parser.add_argument('--weight_decay','-w', type=float, default=5e-4,
                        help='Amount to decay weight at each iteration')
    
    parser.add_argument('--variance_decay', '-v', type=float, nargs=3, metavar=('TARGET','RATE','N'),
                        help='Decay weights such that conv activation variance is TARGET. '
                        'The weights are decayed at RATE every N iterations. '
                        'Good values are  2.25, 0.5, 10.')
    
    parser.add_argument('--weight_redist', '-t', nargs=3, metavar=('NUM','VAR','ITERS'), type=float,
                        help='Rescale weights such that they have the variance VAR. '
                        'Variance is calculated with NUM batches every ITERS and the weights are rescaled according to: '
                        'W[k,:] /= VAR/W[k,:].var() . '
                        'Good values for this are 15,1.5,200, but depend on batchsize and learnrate.'
                        )
    
    parser.add_argument('--augment', '-A', default=False, action='store_true',
                        help='Do data augmentations on the train and test data sets')
    
    parser.add_argument('--old_test_method', default=False, action='store_true',
                        help='Perform accuracy-reducing test augmentations that were previously performed')
    
    parser.add_argument('--seed','-s', type=int, default=None,
                        help='Initial seeding for random numbers')
    
    parser.add_argument('--fast', default=False, action='store_true',
                        help='Run fast training (i.e. special learning rate schedule)')
    
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    
    parser.add_argument('--snapshot_every', metavar='EPOCHS', type=int,
                        help='Number of epochs to take a snapshot after')
    
    parser.add_argument('--update_interval','-u', default=100, type=int,
                        help='Progress bar update interval (iterations)')
    
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    
############################################################################### 
############################################################################### 

class LossHookGradientMethod:
    def add_loss_hook(self, hook, name=None):
        """Registers a loss hook function.

        This function is called on the computed loss.

        Arguments:
        ----------
        hook : function
            Hook function. Must be callable and have the form hook(optimizer, model, loss). 
        name : str
            Name of the registration. If omitted, ``hook.name`` is used by default.

        """
        if not callable(hook):
            raise TypeError('hook function is not callable')

        if name is None:
            name = hook.name
        if name in self._hooks:
            raise KeyError('hook %s already exists' % name)
        self._loss_hooks[name] = hook
    
    
    def remove_hook(self, name):
        """Removes a hook function with specified name"""
        del self._loss_hooks[name]
    
    def call_loss_hooks(self, model, loss):
        """Calls hooks in order on the specified model and loss"""
        for hook in six.itervalues(self._loss_hooks):
            hook(self, model, loss)
    
    def update(self, lossfun=None, *args, **kwds):
        """Updates parameters based on a loss function or computed gradients.

        This method runs in two ways.

        - If ``lossfun`` is given, then it is used as a loss function to
          compute gradients.
        - Otherwise, this method assumes that the gradients are already
          computed.

        In both cases, the computed gradients are used to update parameters.
        The actual update routines are defined by the update rule of each
        parameter.

        """
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', True)
            loss = lossfun(*args, **kwds)
            if use_cleargrads:
                self.target.cleargrads()
            else:
                self.target.zerograds()
            self.call_loss_hooks(self.target, loss) # Added from optimizers.Optimizer
            loss.backward()
            del loss

        self.reallocate_cleared_grads()

        self.call_hooks()

        self.t += 1
        for param in self.target.params():
            param.update()
    

class MomentumSGDLossHooks(LossHookGradientMethod, chainer.optimizers.MomentumSGD):
    """ MomentumSGD with callable hooks to the loss calculation.
    
    These hooks can be used to modify the computation graph or collect statistics
    
    """
        
    def __init__(self, **kwargs):
        super(MomentumSGDLossHooks, self).__init__(**kwargs)
        self._loss_hooks = collections.OrderedDict()
    
class NesterovAGLossHooks(LossHookGradientMethod, chainer.optimizers.NesterovAG):
    """ Nesterov accelerated gradient with callable hooks to the loss calculation.
    
    These hooks can be used to modify the computation graph or collect statistics
    
    """
        
    def __init__(self, **kwargs):
        super(NesterovAGLossHooks, self).__init__(**kwargs)
        self._loss_hooks = collections.OrderedDict()
    

class OrTrigger(object):
    """ Fires if t1 or t2 fires """
    def __init__(self, t1, t2=(1,'epoch')):
        self.t1 = chainer.training.trigger.get_trigger(t1)
        self.t2 = chainer.training.trigger.get_trigger(t2)
    def __call__(self, trainer):
        fire1 = self.t1(trainer)
        fire2 = self.t2(trainer)
        return fire1 or fire2

class PrintReportNoSpecial(extensions.PrintReport):

    """Removes weird control characters from PrintReport """

    def __init__(self, entries, log_report='LogReport', out=sys.stdout):
        super(PrintReportNoSpecial, self).__init__(entries, log_report=log_report, out=out)

    def __call__(self, trainer):
        out = self._out

        if self._header:
            out.write(self._header)
            self._header = None

        log_report = self._log_report
        if isinstance(log_report, str):
            log_report = trainer.get_extension(log_report)
        elif isinstance(log_report, log_report_module.LogReport):
            log_report(trainer)  # update the log report
        else:
            raise TypeError('log report has a wrong type %s' %
                            type(log_report))

        log = log_report.log
        log_len = self._log_len
        while len(log) > log_len:
            self._print(log[log_len])
            log_len += 1
            out.flush()
        self._log_len = log_len
        

def cv_rotate(img, angle):
    # OpenCV doesn't work so well on Cedar
    # scikit-image's rotate function is almost 7x slower than OpenCV
    img = img.transpose(1, 2, 0) / 255.
    img = skimage_transform.rotate(img, angle, mode='edge')
    img = img.transpose(2, 0, 1) * 255.
    img = img.astype('f')
    return img


def transform_fast(inputs, cutout=None, flip=True, crop_size=None):
    """ Stripped down version of the transform function """
    img, label = inputs
    _, H, W = img.shape
    #img_orig = img
    img = img.copy()
    
    # Random flip
    if flip:
        img = transforms.random_flip(img, x_random=True)
        
    if crop_size is not None:
        h0 = np.random.randint(0, H-crop_size[0])
        w0 = np.random.randint(0, W-crop_size[1])
        img = img[:, h0:h0+crop_size[0], w0:w0+crop_size[1]]
    
    if cutout is not None:
        h0, w0 = np.random.randint(0, 32-cutout, size=(2,))
        img[:, h0:h0+cutout, w0:w0+cutout].fill(0.0)
    
    return img, label
    
def transform(inputs, mean=None, std=None, 
              random_angle=15., pca_sigma=255., expand_ratio=1.0,
              crop_size=(32, 32), cutout=None, flip=True, train=True,
              old_test_method=False):
    img, label = inputs
    img = img.copy()

    if train:
        # Random rotate
        if random_angle != 0:
            angle = np.random.uniform(-random_angle, random_angle)
            img = cv_rotate(img, angle)

        # Color augmentation
        if pca_sigma != 0:
            img = transforms.pca_lighting(img, pca_sigma)
            
    elif old_test_method:
        # There was a bug in prior versions, here it is reactivated to preserve
        # the same test accuracy
        if random_angle != 0:
            angle = np.random.uniform(-random_angle, random_angle)
            img = cv_rotate(img, angle)
        

    # Standardization
    if mean is not None:
        img -= mean[:, None, None]
        img /= std[:, None, None]

    if train:
        # Random flip
        if flip:
            img = transforms.random_flip(img, x_random=True)
            
        # Random expand
        if expand_ratio > 1:
            img = transforms.random_expand(img, max_ratio=expand_ratio)
            
        # Random crop
        if tuple(crop_size) != (32, 32) or expand_ratio > 1:
            img = transforms.random_crop(img, tuple(crop_size))
            
        # Cutout
        if cutout is not None:
            h0, w0 = np.random.randint(0, 32-cutout, size=(2,))
            img[:, h0:h0+cutout, w0:w0+cutout].fill(0.0)

    return img, label

def normalize_dataset(dataset, mean, std):
    if not isinstance(dataset, chainer.datasets.TupleDataset):
        raise ValueError('Expected TupleDataset')
    
    old_imgs, labels = dataset._datasets
    mean = mean.astype(old_imgs.dtype)
    std = std.astype(old_imgs.dtype)
    
    if not isinstance(old_imgs, np.ndarray):
        raise ValueError('Expected TupleDataset containing a tuple of numpy arrays')
    
    # Normalization
    imgs = (old_imgs - mean[None, :, None, None]) / std[None, :, None, None]
    
    return chainer.datasets.TupleDataset(imgs, labels)

def pad_dataset(dataset, pad=4):
    """ Pads the dataset using the reflect padding mode """
    imgs, labels = dataset._datasets
    imgs = np.pad(imgs, [(0, 0), (0,0), (pad, pad), (pad, pad)], mode='reflect')
    return chainer.datasets.TupleDataset(imgs, labels)
    
    
def print_header(args, argv, preamble='CIFAR10', printfn=print, 
                 log=open(os.devnull, 'w'),
                 first=('model','dataset','epoch','batchsize','resume','out')):
    """ Prints the arguments and header, and returns a logging print function """
        
    def logprint(*args, file=log, **kwargs):
        if printfn:
            printfn(*args, **kwargs)
        print(*args, file=file, **kwargs)
        file.flush()
    
    vargs = vars(args)
    args_sorted = sorted(vargs.items())
    logprint('{' + ', '.join("'{}':{}".format(k,repr(v)) for k,v, in args_sorted) + '}')
    logprint(' '.join(argv))
    logprint('')
    logprint(preamble)
    logprint('')
    logprint('Arguments: ')
    
    def print_arg(arg):
        logprint('   {:20}: {},'.format("'%s'"%arg,repr(vargs[arg])))
    
    for arg in first:
        print_arg(arg)
    logprint('')
    for arg,_ in args_sorted:
        if arg in first:
            continue
        print_arg(arg)
    
    logprint('')
    
    return logprint
    
def init_model(predictor, class_labels=10, gpu=-1, fast=False):
    """ Initializes the model to train and (optionally) sends it to the gpu """
    if 0: # fast:  # Not sure this helps...
        lossfun = partial(F.softmax_cross_entropy, normalize=False)
    else:
        lossfun = F.softmax_cross_entropy
        
    model = L.Classifier(predictor(class_labels), lossfun=lossfun)
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()  # Make the GPU current
        model.to_gpu()
        
    return model

def model_from_snapshot(predictor, snapshot):
    chainer.serializers.load_npz(snapshot, predictor, path='updater/model:main/')

def model_deepcopy(model_to_copy):
    ''' Returns a deep copy of a model (no shared params) '''
    return copy.deepcopy(model_to_copy)

def _linear_congruential_rng(seed):
    ''' glibc's linear congruential generator'''
    while True:
        seed = int( (seed*1103515245 + 12345)%(2**31-1) ) if seed is not None else None
        yield seed
        
def seed_rng(seed, gpu=-1):
    if seed is None:
        return None, None
    
    os.environ['CHAINER_SEED'] = str(seed)
    
    rng = _linear_congruential_rng(seed)
    next(rng) # Make sure we don't use the same seed as for main chainer rng
        
    fixed_seeds = (next(rng), next(rng))
    if gpu >= 0 and seed is not None:
        reseed_rng(fixed_seeds)
    
    return rng, fixed_seeds

def reseed_rng(fixed_seeds):
    if fixed_seeds is not None:
        cuda.cupy.random.seed(fixed_seeds[0])
        cuda.numpy.random.seed(fixed_seeds[1])
    

def load_dataset(batchsize, dataset, augment=False, fast=False, old_test_method=False):
        
    scale = 255.0 if augment else 1.0
    if dataset == 'cifar10':
        train, test = get_cifar10(scale=scale)
        class_labels = 10
    elif dataset == 'cifar100':
        train, test = get_cifar100(scale=scale)
        class_labels = 100
    else:
        raise RuntimeError('Invalid dataset choice.')
    
    
    if augment:
        #mean = np.mean(train._datasets[0], axis=(0, 2, 3))
        #std = np.std(train._datasets[0], axis=(0, 2, 3))
        # Pre calculated from above
        mean = np.array([ 125.30690002,  122.95014954,  113.86599731])
        std = np.array([ 62.9932518,   62.08860397,  66.70500946])
        
        train = normalize_dataset(train, mean, std)
        test = normalize_dataset(test, mean, std)
        
        # Previously pca was 25.5 or 10% of 255
        # Now we normalize, so to keep PCA at 10% of the range we use the min and max of the 
        # normalized datasets
        
        #pca_sigma = 0.2 * (np.max(train._datasets[0] - np.min(train._datasets[0])
        # Pre calculated from above
        pca_sigma = 0.1 * ((2.126797) - (-1.9892114))  # = 0.4116
        
        slow_augment = dict(crop_size=(32, 32), expand_ratio=1.2, 
                            pca_sigma=pca_sigma, random_angle=15.0, 
                            train=True)
        fast_augment = dict(crop_size=(32, 32), cutout=8, flip=True)
        
        
        if fast:
            train = pad_dataset(train, pad=4)
            train_transform = partial(transform_fast, **fast_augment)
            test_transform = lambda x:x # No augmentation
        else:
            train_transform = partial(transform, **slow_augment)
            test_transform = partial(transform, train=False, old_test_method=old_test_method)
    
        train = TransformDataset(train, train_transform)
        test = TransformDataset(test, test_transform)
    
    
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)
    
    return train_iter, test_iter, class_labels

class PiecewiseLinearShift(extensions.LinearShift):
    """ A linearly shifts piecewise through the specified epochs
    
    if value_range is (0, 1, -1) and time_range is (2, 4, 6)
    then the values at each epoch will be:
        
        epoch = [ 0,     1,   2,   3,   4,   5,    6]
        value = [ 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, -1.0]
    """
    
    def __init__(self, attr, value_range, time_range, optimizer=None):
        super(PiecewiseLinearShift, self).__init__(attr, value_range, time_range, optimizer=optimizer)
        if len(value_range) != len(time_range):
            raise Exception('Value and time range lengths are different!')
        self._time_range = np.array(self._time_range)
    
    def _compute_next_value(self):
        """ Overrides simple LinearShift next value with a piecewise version """
        return np.interp(self._t, self._time_range, self._value_range)

def get_lr_schedule(args, train_iter, fast=False, peak_epoch=5):
    epochs = args.epoch
    lr = args.learnrate
    decay = args.learnrate_decay
    
    if not fast:
        # Exponential decay
        initial_lr = lr
        lr_ext = extensions.ExponentialShift('lr', 0.5)
        lr_trigger = (decay, 'epoch')
    
    else:
        # Linear ramp up and down
        iters_per_epoch = len(train_iter.dataset)/train_iter.batch_size
        iters = int(epochs * iters_per_epoch)
        peak_iter = int(peak_epoch / epochs * iters)
        
        # lr = lr/batchsize # original work claims to decrease learning rate by batch size
        initial_lr = lr / peak_iter
        lr_ext = PiecewiseLinearShift('lr', (initial_lr, lr, 0.0), (1, peak_iter, iters))
        lr_trigger = (1, 'iteration')
        
         # Unused, better to do a per-iteration change
        '''
        initial_lr = lr / peak_epoch
        lr_ext = PiecewiseLinearShift('lr', (initial_lr, lr, 0.0), (0, peak_epoch-1, epochs))
        lr_trigger = (1, 'epoch')
        '''
        
    return initial_lr, lr_ext, lr_trigger

if __name__ == '__main__':
    main()
