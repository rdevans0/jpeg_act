import heapq

import numpy as np

# Conditional import of get_array_module if cupy (GPU numpy) is available
try:
    import cupy
    from cupy import get_array_module
    cupy_available = True
except ImportError:
    cupy = None
    get_array_module = lambda *a: np
    cupy_available = False



# USEFUL ITERATORS
def backward_var_iter(start):
    ''' An iterator for going down the backprop chain '''
    cand_funcs = []
    seen = set()
    
    def add_cand(cand):
        if cand not in seen:
            # Negate since heapq is min-heap
            heapq.heappush(cand_funcs, (-cand.rank, len(seen), cand))
            seen.add(cand)
            
    add_cand(start.creator_node)
    
    while cand_funcs:
        rank, _, func = heapq.heappop(cand_funcs)
        inputs = func.inputs
        target_inputs = [x for x in inputs if x.requires_grad]
        if not target_inputs:
            continue
        for x in target_inputs:
            if x.creator_node is not None:
                yield (-rank, func, x)
                add_cand(x.creator_node)


def backward_var_iter_nodup(start):
    """ Same as backward_var_iter, with no duplicate variables (by ID) """
    seen = set()
    seen.add(id(start))
    
    for rank,func,var in backward_var_iter(start):
        if id(var) in seen:
            continue
        yield (rank,func,var)
        seen.add(id(var))

def backward_func_iter_nodup(start):
    ''' An iterator for going down the backprop chain '''
    cand_funcs = []
    seen = set()
    
    def add_cand(cand):
        if cand not in seen:
            # Negate since heapq is min-heap
            heapq.heappush(cand_funcs, (-cand.rank, len(seen), cand))
            seen.add(cand)
            
    add_cand(start.creator_node)
    
    while cand_funcs:
        rank, _, func = heapq.heappop(cand_funcs)
        inputs = func.inputs
        target_inputs = [x for x in inputs if x.requires_grad]
        
        for x in target_inputs:
            if x.creator_node is not None:
                add_cand(x.creator_node)
        
        yield(func)

