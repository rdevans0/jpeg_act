from __future__ import print_function
import argparse
import textwrap
import re
import os
import sys
   

import chainer
from chainer import training
from chainer.training import extensions

from chainer import cuda

DEFAULT_LAYERS = None #!TODO: Remove global

from train_cifar import models
from train_cifar import add_base_args
from train_cifar import seed_rng
from train_cifar import load_dataset
from train_cifar import init_model
from train_cifar import print_header
from train_cifar import get_lr_schedule
from train_cifar import PrintReportNoSpecial
from train_cifar import MomentumSGDLossHooks


from common.utils import backward_var_iter_nodup
from common.helpers import JPEGHelper, FixPointHelper, Fix35Helper

from functools import partial

def main(argv=sys.argv[1:]):
    
    if type(argv) == str:
        argv = argv.split()
    
    parser = ArgumentParserWithEpilog(
            description='Chainer CIFAR with recall error:')
    
    # Command line arguments
    add_base_args(parser)
    parser.add_argument('--dynamic_rescale', '-R', default=False, type=float,
                        help='Rescale activations to this range [-R,+R] on a per-channel basis, before compressing')
    add_ae_args(parser)
    args = parser.parse_args(argv)
    
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
    print_log = print_header(args, argv, log=report, preamble='CIFAR10/100 (%s)'%__file__)
    
    ##
    # Set up model and dataset iterators
    rng, fixed_seeds = seed_rng(args.seed, args.gpu)   
    train_iter, val_iter, class_labels = load_dataset(args.batchsize, args.dataset, 
                                                      args.augment, args.fast,
                                                      args.old_test_method)
    model = init_model(models[args.model], class_labels=class_labels, gpu=args.gpu, fast=args.fast)
    
    ## 
    # Get the recall error helper map
    all_layers = model.predictor.act_names
    helper_map, filterspec_map = parse_ae_args(parser, args, rng, all_layers=all_layers)
    print_helper_summary(helper_map, filterspec_map, print_log)
    print_helper_map(all_layers, helper_map, print_log)
    
    # Set up an optimizer
    lr, lr_ext, lr_trigger = get_lr_schedule(args, train_iter, fast=args.fast)
    
    optimizer = MomentumSGDScrambler(helper_map, compress_x_hat=False,
                                     dynamic_rescale=args.dynamic_rescale,
                                     lr=lr, momentum=args.momentum)
    optimizer.setup(model)
    
    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, end_trigger, out=args.out)
    
    # Decay
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
    
    # Learning rate schedule
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
        #chainer.serializers.load_npz(args.resume, trainer)
        from train_cifar import model_from_snapshot
        model_from_snapshot(model, args.resume)
        
    
    ##
    # Run the training
    trainer.run()
    
    report.close()
    
    return trainer, None, helper_map



###############################################################################
###############################################################################

# Abstract updater class
class ScramblerUpdater:
    def scrambler_init(self, scrambler_map, compress_x_hat, dynamic_rescale):
        self.scrambler_map = scrambler_map
        self.compress_x_hat = compress_x_hat
        self.dynamic_rescale = dynamic_rescale
    
    def update(self, lossfun=None, *args, **kwds):
        ''' Modified from class GradientMethod(Optimizer) '''
        
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', True)
            loss = lossfun(*args, **kwds)
            if use_cleargrads:
                self.target.cleargrads()
            else:
                self.target.zerograds()
                
            if hasattr(self, 'call_loss_hooks'):
                self.call_loss_hooks(self.target, loss)
                
            
            # SCRAMBLE IT!
            for scrambler in set(self.scrambler_map.values()):
                if scrambler is not None:
                    scrambler.scramble_setup()
                
            for rank,func,var in backward_var_iter_nodup(loss):
                if var.data is None:
                    continue
                scrambler = self.scrambler_map[var.name]
                if scrambler is not None:
                    if self.dynamic_rescale:
                        sf = dynamic_scale(var, self.dynamic_rescale)
                        
                    scrambler.scramble(rank, func, var)
                    
                    if self.dynamic_rescale:
                        dynamic_unscale(var, sf)
                
                    if self.compress_x_hat and func.label == 'BatchNormalizationFlexFunc':
                        if var.name and var.name.endswith('-c'):
                            scrambler.ascramble(func.x_hat)
                            var.data *= cuda.numpy.NaN # This variable *SHOULDNT* be used
            
            self.target.predictor.post_scramble_callback(loss, self.scrambler_map)
            
            loss.backward()
            del loss

        self.reallocate_cleared_grads()
        self.call_hooks()
        
        self.t += 1
        for param in self.target.params():
            param.update()

class MomentumSGDScrambler(ScramblerUpdater, MomentumSGDLossHooks):
    def __init__(self, scrambler_map, compress_x_hat=False, dynamic_rescale=False, **kwargs):
        super(MomentumSGDScrambler, self).__init__(**kwargs)
        self.scrambler_init(scrambler_map, compress_x_hat, dynamic_rescale)
        
def dynamic_scale(var, scale_target, in_place=True, eps=1e-5):
    data = var.data if in_place else var
    
    x_max = abs(data).max(axis=[0,2,3])
    #x_max[x_max == 0] = 1.0 # Corrects for INFs when divided by this
    scale_factor = (scale_target / (x_max + eps) ) [None, :, None, None]
    
    if in_place:
        var.data *= scale_factor
        return scale_factor
    else:
        return data*scale_factor, scale_factor

def dynamic_unscale(var, scale_factor, in_place=True):
    if in_place:
        var.data *= 1./scale_factor
    else:
        return var * (1./scale_factor)
    

###############################################################################
###############################################################################
                
def apply_helper_to_map(helper, helper_map, filters, parser, all_layers=DEFAULT_LAYERS):
    
    expanded = dict((v[0],v) for v in ['c','r','p','n','f','d','s','b','x','i'])
    def select(layers, filt):
        for l in layers:
            if helper_map[l] not in (False,None) and helper_map[l] != helper:
                parser.error('The filter {} overlaps with a previous filter at layer {}'.format(filt, l))
            helper_map[l] = helper
    
    for filt in filters.split(','):
        #!TODO How to match this better???
        m = re.findall('([ABFOWR]+)(\d+)?(?:_(\d+))?(?:_(\d+))?([crpnfdsbxi]*)', filt)
        if len(m) != 1:
            parser.error('Filter "{}" is malformed at {}'.format(filters, filt))
        
        category, groups, subgroups, subsubgroups, layer_types = m[0]
        
        # Make all items unique and sort them
        category, groups, subgroups, subsubgroups, layer_types =\
                map(lambda x: sorted(set(x)), 
                    (category, groups, subgroups, subsubgroups, layer_types))
                
        # Closer to regex format
        groups = ''.join(groups)
        subgroups = ''.join(subgroups)
        subsubgroups = ''.join(subsubgroups)
        layer_types = '|'.join(expanded[t] for t in layer_types)
        
        if 'R' in category and 'W' in category:
            parser.error('R and W cannot be used together. Specify blocks in Resnet and Widenet, respectively')
        
        # Do the selection
        if 'A' in category:
            if groups or subgroups or layer_types: 
                parser.error('A can only be specified on its own (e.g. A3f doesnt work)')
            select(all_layers, filt)
            
        if 'O' in category:
            select(filter(lambda l: l == None, all_layers), filt)
        
        layers = list(filter(lambda l: l != None, all_layers))
        
        if 'F' in category:
            expr = 'final'
            expr += '[%s]'%groups if groups else '\d'
            if layer_types:
                expr += '-(%s)'%layer_types
            select(filter(lambda l: re.match(expr, l), layers), filt)
        
        if 'B' in category:
            expr = 'block'
            expr += '[%s]'%groups if groups else '\d'
            expr += '_'
            expr += '[%s]'%subgroups if subgroups else '\d'
            if layer_types:
                expr += '-(%s)'%layer_types
            select(filter(lambda l: re.match(expr, l), layers), filt)
        
        if 'W' in category or 'R' in category:
            expr = 'wide' if 'W' in category else 'res'
            expr += '[%s]'%groups if groups else '\d'
            expr += '_'
            expr += '[%s]'%subgroups if subgroups else '\d+'
            expr += '_'
            expr += '[%s]'%subsubgroups if subsubgroups else '\d'
            if layer_types:
                expr += '-(%s)'%layer_types
            print(filt, expr)
            select(filter(lambda l: re.match(expr, l), layers), filt)
    
    if sum(v == helper for v in helper_map.values()) == 0:
        parser.error('Filter {} selected no layers!'.format(filters))
    
    return helper_map

def print_helper_summary(helper_map, filterspec_map, printfn=print):
    for helper, filterspec in filterspec_map.items():
        count = sum(v == helper for v in helper_map.values())
        printfn('# applied error: {0:<15}\t{1} ({2})'.format(helper.name, filterspec, count))
        helper.print_settings(printfn)

def print_helper_map(sorted_layers, helper_map, printfn=print):
    """ Nicely prints the helper_map using printfn. Order specified by sorted_layers """
    
    
    printfn('\n# Error Map')
              
    if all(v is None for v in helper_map.values()):
        printfn('#  No layers selected!')
        return
    
    # Cluster names according to prefix
    clustered = [ [name if name is not None else 'None-'] for name in reversed(sorted_layers)]
    head = 0
    get_prefix = lambda n: n.split('-')[0]
    get_postfix = lambda n: '-'.join(n.split('-')[1:])
    while head < len(clustered):
        prefix = get_prefix(clustered[head][0])
        tail = head + 1
        while tail < len(clustered):
            name = clustered[tail][0]
            if get_prefix(name) != prefix:
                break
            clustered[head] += [name]
            clustered.pop(tail)
        head = head+1
    
    for cluster in clustered:
        
        if cluster == ['None-']:
            cluster = [None]
        
        if len(cluster) == 1:
            layer = cluster[0]
            h = helper_map[layer]
            printfn('#   {0: <16} -> {1}'.format(str(layer), ' -- ' if h is None else h.detail))
            continue
        
        statuses = [' -- ' if helper_map[layer] is None else helper_map[layer].detail for layer in cluster]
        
        if all( status == statuses[0] for status in statuses ):
            prefix = get_prefix(cluster[0])
            postfix = '-[' + ''.join(get_postfix(layer) for layer in cluster) +']'
            printfn('#   {0: <16} -> {1}'.format(prefix + postfix, statuses[0]))
        else:
            for layer in cluster:
                h = helper_map[layer]
                printfn('#   {0: <16} -> {1}'.format(str(layer), ' -- ' if h is None else h.detail))
            

###############################################################################
###############################################################################
                
                
def add_ae_args(parser, 
                include=('jpeg',
                         'fixpoint',
                         'fix35',
                         )):
    
    if 'jpeg' in include:
        parser.add_argument('--ae_jpeg', nargs=2, metavar=('dqt','filters'), action='append',
                    help='Jpeg error activation function. dqt specifies the quantization matrix. '+\
                    'Note that this implementation scales by 32x and truncates to 8 bits. ' +\
                    'Filters specifies which layers to apply to the method to (see <filters>).')
    
    if 'fixpoint' in include:
        parser.add_argument('--ae_fixpoint', nargs=3, metavar=('whole_bits','frac_bits','filters'), action='append',
                    help='Fixpoint rounding activation function. Rounds to a representation with '+\
                    'whole_bits+frac_bits total bits, with a decimal point at frac_bits. Automatically handles signed and unsigned. ' +\
                    'Filters specifies which layers to apply to the method to (see <filters>).')
    
    if 'fix35' in include:
        parser.add_argument('--ae_fix35', nargs=1, metavar=('filters'), action='append',
                    help='Fixpoint rounding activation function. Rounds to a representation with '+\
                    '8 bits: 3.5. Automatically handles signed and unsigned. ' +\
                    'Filters specifies which layers to apply to the method to (see <filters>).')
        
        

def parse_ae_args(parser, args, rng_seeds, all_layers = DEFAULT_LAYERS):
    """ Decode the helpers for each argument and put them into a map of layer->helper or None"""
    
    helper_classes = {
        'jpeg': JPEGHelper,
        'fixpoint': FixPointHelper,
        'fix35': Fix35Helper,
    }
    
    helper_map = dict( (k,None) for k in all_layers)
    filterspec_map = {}
    
    for helper_type in helper_classes.keys():
        ae_args_all = getattr(args, 'ae_'+helper_type)
        if ae_args_all is None:
            continue  # Argument was not set
        for ae_args in ae_args_all:
            helper_args = ae_args[:-1]
            filters = ae_args[-1]
            
            h = helper_classes[helper_type](*helper_args)
            
            filterspec_map[h] = filters
            apply_helper_to_map(h, helper_map, filters, parser, all_layers=all_layers)
            
    return helper_map, filterspec_map


###############################################################################
###############################################################################

class WrapFormatter(argparse.RawTextHelpFormatter):
    def _split_lines(self, text, width):
        sp = text.splitlines()
        return sum((textwrap.wrap(t, width) if t else [' '] for t in sp), [])

FILTER_FORMAT_DESC = textwrap.dedent('''\
        Layer Filters <filters>:
        
            This is a single, or comma separated list of filters. 
            Each filter can specify one or more layers.
            
            The syntax is (in python regex format):
            
                [ABFO]+(\d+)?(_\d)?[crpnfd]*
                
                [ABFO]+     Group categories (all, block, final, other)
                (\d+)?      Group number(s) (omitting selects all groups)
                (_\d)?      Subgroup number (omitting this selects the entire group[s])
                [crpnfd]*   Layer types:
                                c - convolutional
                                r - relu
                                p - 2d max pooling
                                n - batch normalization
                                f - fully connected
                                d - dropout
            
            Examples (* is wildcard):
                
                A           *
                B3c         block3*-cv (convolutional)
                B123_1      block1_1-*, block2_1-*, and block3_1-*
                B_1r        block*_1-relu (if available)
                F1f         final1-fc (fully connected)
                F           final*
                B           block*
                B1rc        block1_*-relu and block1_*-cv
                Bc,B1_1r    block*-conv and block1_1-relu
             
        ''')

ArgumentParserWithEpilog = partial(argparse.ArgumentParser,
                                     formatter_class=WrapFormatter,
                                     epilog=FILTER_FORMAT_DESC)

###############################################################################
###############################################################################
        


if __name__ == '__main__':
    main()
