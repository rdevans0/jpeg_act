
from chainer import cuda
import numpy as np
import argparse
from heapq import heappush, heappop, heapify # Huffman encoder

from common.helpers import Helper

# QUANTIZATION MATRICES

QUANT_TABLES = {
    
    'test1':np.array([ # kinda like jpeg50
          1,  6, 10, 20, 32, 45, 59, 89,
          3, 15, 38, 49, 58, 67, 91, 96,
          8, 37, 45, 54, 63, 87, 94, 98,
         12, 48, 55, 63, 86, 92, 97,100,
         27, 59, 65, 87, 92, 96,100,102,
         45, 70, 89, 93, 97,100,103,104,
         61, 93, 96, 99,101,103,104,105,
         91, 98,100,102,103,105,105,106, 
        ]),

    'test2':np.array([ # kinda like jpeg80
          1,  1,  1,  1,  1,  7, 14, 24,
          1,  3,  4,  9, 14, 18, 26, 29,
          1,  3,  7, 12, 16, 23, 28, 31,
          1,  9, 12, 16, 22, 26, 30, 33,
          1, 14, 17, 23, 26, 30, 32, 34,
          7, 20, 24, 27, 30, 32, 34, 35,
         15, 27, 29, 31, 33, 35, 36, 36,
         26, 31, 32, 34, 35, 36, 36, 37,
        ]),
    
    'jpec': np.array([
        	16, 11, 10, 16, 24, 40, 51, 61,
        	12, 12, 14, 19, 26, 58, 60, 55,
        	14, 13, 16, 24, 40, 57, 69, 56,
        	14, 17, 22, 29, 51, 87, 80, 62,
        	18, 22, 37, 56, 68,109,103, 77,
        	24, 35, 55, 64, 81,104,113, 92, # 35->36 vs std
        	49, 64, 78, 87,103,121,120,101,
        	72, 92, 95, 98,112,100,130, 99  # 130 ->103 vs std
        ]),

    'jpeg98': np.array([
          1,  1,  1,  1,  1,  2,  2,  2,
          1,  1,  1,  1,  1,  2,  2,  2,
          1,  1,  1,  1,  2,  2,  3,  2,
          1,  1,  1,  1,  2,  3,  3,  2,
          1,  1,  1,  2,  3,  4,  4,  3,
          1,  1,  2,  3,  3,  4,  5,  4,
          2,  3,  3,  3,  4,  5,  5,  4,
          3,  4,  4,  4,  4,  4,  4,  4,
         ]),

    'jpeg95': np.array([ # jpeg95  7.03x-28.2x CR (mean = 12.0x)
          2,  1,  1,  2,  2,  4,  5,  6,
          1,  1,  1,  2,  3,  6,  6,  6,
          1,  1,  2,  2,  4,  6,  7,  6,
          1,  2,  2,  3,  5,  9,  8,  6,
          2,  2,  4,  6,  7, 11, 10,  8,
          2,  4,  6,  6,  8, 10, 11,  9,
          5,  6,  8,  9, 10, 12, 12, 10,
          7,  9, 10, 10, 11, 10, 10, 10,
         ]),

    'jpeg90': np.array([ # jpeg90  10.0x-42.0x CR (mean = 17.6x)
          3,  2,  2,  3,  5,  8, 10, 12,
          2,  2,  3,  4,  5, 12, 12, 11,
          3,  3,  3,  5,  8, 11, 14, 11,
          3,  3,  4,  6, 10, 17, 16, 12,
          4,  4,  7, 11, 14, 22, 21, 15,
          5,  7, 11, 13, 16, 21, 23, 18,
         10, 13, 16, 17, 21, 24, 24, 20,
         14, 18, 19, 20, 22, 20, 21, 20   
         ]),

    'jpeg80' : np.array([ # jpeg80  14.5x-75.6x CR (mean = 29.1x)
          6,  4,  4,  6, 10, 16, 20, 24,
          5,  5,  6,  8, 10, 23, 24, 22,
          6,  5,  6, 10, 16, 23, 28, 22,
          6,  7,  9, 12, 20, 35, 32, 25,
          7,  9, 15, 22, 27, 44, 41, 31,
         10, 14, 22, 26, 32, 42, 45, 37,
         20, 26, 31, 35, 41, 48, 48, 40,
         29, 37, 38, 39, 45, 40, 41, 40
        ]),

    'jpeg': np.array([    # jpeg50  22.9x-116.6x CR (mean = 51.6x)
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68,109,103, 77,
        24, 36, 55, 64, 81,104,113, 92,
        49, 64, 78, 87,103,121,120,101,
        72, 92, 95, 98,112,100,103, 99
        ]), # https://web.stanford.edu/class/ee398a/handouts/lectures/08-JPEG.pdf

    'sym1':  np.array([
          1,  3,  7,  9, 11, 13, 15, 17, 
          3,  7,  9, 11, 13, 15, 17, 19,  
          7,  9, 11, 13, 15, 17, 19, 21,
          9, 11, 13, 15, 17, 19, 21, 23, 
         11, 13, 15, 17, 19, 21, 23, 25,
         13, 15, 17, 19, 21, 23, 25, 27,
         15, 17, 19, 21, 23, 25, 27, 29,
         17, 19, 21, 23, 25, 27, 29, 31   
        ]),
    
    'pil': np.array([
         16, 11, 12, 14, 12, 10, 16, 14,
         13, 14, 18, 17, 16, 19, 24, 40,
         26, 24, 22, 22, 24, 49, 35, 37,
         29, 40, 58, 51, 61, 60, 57, 51,
         56, 55, 64, 72, 92, 78, 64, 68,
         87, 69, 55, 56, 80,109, 81, 87,
         95, 98,103,104,103, 62, 77,113,
        121,112,100,120, 92,101,103, 99,        
        ]), # Pulled from PIL encoded file under the quantization table section
   '123_1': np.array([
          1,  3,  3,  3,  3,  3,  3,  3,
          3,  2,  2,  2,  2,  2,  2,  2,
          3,  2,  2,  2,  2,  2,  2,  2,
          3,  2,  2,  2,  2,  2,  2,  2,
          3,  2,  2,  2,  2,  2,  2,  2,
          3,  2,  2,  2,  2,  2,  2,  2,
          3,  2,  2,  2,  2,  2,  2,  2,
          3,  2,  2,  2,  2,  2,  2,  2,    
        ]),

    'ones': np.ones((64,)),
    
    
    'optL': np.array([
         8,   7,   7,   7,   7,   6,   6,   6,  # orig first element was 9
         8,   7,   7,   7,   6,   6,   6,   6,
         7,   7,   6,   6,   6,   6,   6,   6,
         7,   6,   6,   6,   6,   6,   5,   5,
         7,   6,   6,   6,   6,   6,   5,   5,
         7,   6,   6,   6,   6,   6,   5,   5,
         7,   6,   6,   6,   6,   5,   5,   5,
         6,   6,   6,   6,   5,   5,   5,   5,
        ]),
    'optsL': np.array([  # (2**np.rint(np.log2(optL))).astype('i')
         8,   8,   8,   8,   8,   8,   8,   8,
         8,   8,   8,   8,   8,   8,   8,   8,
         8,   8,   8,   8,   8,   8,   8,   8,
         8,   8,   8,   8,   8,   8,   8,   8,
         8,   8,   8,   8,   8,   8,   8,   8,
         8,   8,   8,   8,   8,   8,   8,   8,
         8,   8,   8,   8,   8,   8,   8,   8,
         8,   8,   8,   8,   8,   8,   8,   8,
        ]),  
    
    'optH': np.array([
         8,  26,  25,  25,  24,  24,  23,  22,  # orig first element was 29
        26,  24,  24,  23,  22,  22,  21,  20,
        26,  24,  24,  23,  22,  21,  20,  20,
        25,  23,  23,  22,  21,  21,  20,  19,
        25,  23,  22,  21,  21,  20,  19,  19,
        24,  22,  21,  20,  20,  20,  19,  19,
        23,  21,  20,  20,  19,  19,  19,  19,
        22,  20,  20,  19,  19,  19,  19,  19,
        ]),  
    
    'optsH': np.array([ # (2**np.rint(np.log2(optL))).astype('i')
         8,  32,  32,  32,  32,  32,  32,  16,
        32,  32,  32,  32,  16,  16,  16,  16,
        32,  32,  32,  32,  16,  16,  16,  16,
        32,  32,  32,  16,  16,  16,  16,  16,
        32,  32,  16,  16,  16,  16,  16,  16,
        32,  16,  16,  16,  16,  16,  16,  16,
        32,  16,  16,  16,  16,  16,  16,  16,
        16,  16,  16,  16,  16,  16,  16,  16,
        ]),  
    
    '8_16': np.array([  # All 16's, except for element 0
        8,   16,  16,  16,  16,  16,  16,  16,
        16,  16,  16,  16,  16,  16,  16,  16,
        16,  16,  16,  16,  16,  16,  16,  16,
        16,  16,  16,  16,  16,  16,  16,  16,
        16,  16,  16,  16,  16,  16,  16,  16,
        16,  16,  16,  16,  16,  16,  16,  16,
        16,  16,  16,  16,  16,  16,  16,  16,
        16,  16,  16,  16,  16,  16,  16,  16,
        ]),  
}

# ZIG ZAG ORDER
# order is (0,0), (0,1),(1,0), (2,0),(1,1),(0,2), ... assuming an 8x8 block
# These are linear indices
# from gen_rle_order(do_print=True)
JPEC_ZZ_ENC = np.array([
      0,  1,  8, 16,  9,  2,  3, 10,
     17, 24, 32, 25, 18, 11,  4,  5,
     12, 19, 26, 33, 40, 48, 41, 34,
     27, 20, 13,  6,  7, 14, 21, 28,
     35, 42, 49, 56, 57, 50, 43, 36,
     29, 22, 15, 23, 30, 37, 44, 51,
     58, 59, 52, 45, 38, 31, 39, 46,
     53, 60, 61, 54, 47, 55, 62, 63
]).astype('i')


def print_dqt(dqt, pre='  ', post=''):
    for line in dqt.reshape((8,8)):
        print(pre + ','.join('%4d'%v for v in line) + post)

def banded_dqt(name):
    bands = name.replace('band','').split(',')
    if len(bands) is not 4:
        raise Exception('Invalid band format {}'.format(name))
    
    diags = np.array([
          1,  2,  3,  4,  5,  6,  7,  8, 
          2,  3,  4,  5,  6,  7,  8,  9,  
          3,  4,  5,  6,  7,  8,  9, 10,
          4,  5,  6,  7,  8,  9, 10, 11, 
          5,  6,  7,  8,  9, 10, 11, 12,
          6,  7,  8,  9, 10, 11, 12, 13,
          7,  8,  9, 10, 11, 12, 13, 14,
          8,  9, 10, 11, 12, 13, 14, 15,   
        ])
    
    lf = diags <= 3
    mf = (diags > 3) * (diags <= 8)
    hf = (diags > 8) * (diags <= 12)
    uhf = 1 - lf - mf - hf
    
    k = [int(band) for band in bands]
    
    dqt = lf*k[0] + mf*k[1] + hf*k[2] + uhf*k[3]
    dqt = np.clip(dqt, 1, 255)
    return dqt

def jpeg_dqt(quality):
    JPEG_STD_QZR = np.array([
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68,109,103, 77,
        24, 36, 55, 64, 81,104,113, 92,
        49, 64, 78, 87,103,121,120,101,
        72, 92, 95, 98,112,100,103, 99
    ]) # https://web.stanford.edu/class/ee398a/handouts/lectures/08-JPEG.pdf
    
    
    scale = 50./quality if quality < 50  else 2 - quality/50.
    a = (JPEG_STD_QZR[:64].astype('f')*scale + 0.5).astype('i')
    a = a.clip(1,255)
    return a.reshape((8,8))

def quant_enc_cupy(dct, dqt):
    h,w = dct.shape
    # Note linear indices are m*8+n
    cuda.elementwise(
        'raw float32 dqt, int32 w, int32 h',
        'float32 quant',
        '''
            int m,n,mm,nn,k,tmp;
            
            m = i/w;
            n = i%w;
            mm = m%8;
            nn = n%8;
            k = mm*8 + nn;              // Linear index into 8x8 block
            tmp = (int) (quant / dqt[k]); // Quantization
            quant = (float) tmp;
        
        ''',
        'quant_2d'
        ) (dqt, w, h, dct)
    return dct

def quant_dec_cupy(dct, dqt):
    h,w = dct.shape
    # Note linear indices are m*8+n
    cuda.elementwise(
        'raw float32 dqt, int32 w, int32 h',
        'float32 quant',
        '''
            int m,n,mm,nn,k;
            float tmp;
            
            m = i/w;
            n = i%w;
            mm = m%8;
            nn = n%8;
            k = mm*8 + nn;              // Linear index into 8x8 block
            tmp = (quant * dqt[k]);       // De Quantization
            quant = (float) tmp;
        
        ''',
        'quant_2d'
        ) (dqt, w, h, dct)
    return dct

def dct_cupy_2dx(img, bias=0):
    
    h,w = img.shape
    dct = cuda.cupy.empty_like(img)
    # Note linear indices are m*8+n
    cuda.elementwise(
        'raw float32 img, int32 w, int32 h, int32 bias',
        'float32 dct',
        '''
            const float coeff[7] = {0.4903926, 0.4619398, 0.4157348, 0.3535534, 0.2777851, 0.1913417, 0.0975452};
            float tmp[8];
            float s0,s1,s2,s3,d0,d1,d2,d3;
            
            int m = i/w;
            int n = i%w;
            int bn = n/8;           // block number in x
            int start = m*w + bn*8; // start index of x block
            s0 = img[start + 0*8 + 0] + img[start + 0*8 + 7];
            s1 = img[start + 0*8 + 1] + img[start + 0*8 + 6];
            s2 = img[start + 0*8 + 2] + img[start + 0*8 + 5];
            s3 = img[start + 0*8 + 3] + img[start + 0*8 + 4];
            
            d0 = img[start + 0*8 + 0] - img[start + 0*8 + 7];
            d1 = img[start + 0*8 + 1] - img[start + 0*8 + 6];
            d2 = img[start + 0*8 + 2] - img[start + 0*8 + 5];
            d3 = img[start + 0*8 + 3] - img[start + 0*8 + 4];
            
            // Returns the correct dct coefficient for the element
            tmp[0] = coeff[3]*(s0+s1+s2+s3);
            tmp[1] = coeff[0]*d0+coeff[2]*d1+coeff[4]*d2+coeff[6]*d3;
            tmp[2] = coeff[1]*(s0-s3)+coeff[5]*(s1-s2);
            tmp[3] = coeff[2]*d0-coeff[6]*d1-coeff[0]*d2-coeff[4]*d3;
            tmp[4] = coeff[3]*(s0-s1-s2+s3);
            tmp[5] = coeff[4]*d0-coeff[0]*d1+coeff[6]*d2+coeff[2]*d3;
            tmp[6] = coeff[5]*(s0-s3)+coeff[1]*(s2-s1);
            tmp[7] = coeff[6]*d0-coeff[4]*d1+coeff[2]*d2-coeff[0]*d3;
            
            dct = tmp[n%8];
        
        ''',
        'dct_2d'
        ) (img, w, h, bias,  dct)
    return dct

def idct_cupy_2dx(dct):
    ''' Should get the job done, based on the inverse matrix from https://unix4lyfe.org/dct-1d/'''
    h,w = dct.shape
    img = cuda.cupy.empty_like(dct)
    # Note linear indices are m*8+n
    

  
    cuda.elementwise(
        'raw float32 dct, int32 w, int32 h',
        'float32 img',
        '''
            const float coeff[7] = {0.4903926, 0.4619398, 0.4157348, 0.3535534, 0.2777851, 0.1913417, 0.0975452};
            float tmp[8];
            float a,b,c,d,e,f,g,h;
            float c0,c1,c2,c3,c4,c5,c6,c7;
            
            int m = i/w;
            int n = i%w;
            int bn = n/8;           // block number in x
            int start = m*w + bn*8; // start index of x block
            int bias = 0;
            
            c0 = coeff[3];
            c1 = coeff[0];
            c2 = coeff[1];
            c3 = coeff[2];
            c4 = coeff[3];
            c5 = coeff[4];
            c6 = coeff[5];
            c7 = coeff[6];
            
            a = dct[start + 0*8 + 0];
            b = dct[start + 0*8 + 1];
            c = dct[start + 0*8 + 2];
            d = dct[start + 0*8 + 3];
            e = dct[start + 0*8 + 4];
            f = dct[start + 0*8 + 5];
            g = dct[start + 0*8 + 6];
            h = dct[start + 0*8 + 7];
            
            
            // Unfactored
            /*
            tmp[0] = c0*a + c1*b + c2*c + c3*d + c4*e + c5*f + c6*g + c7*h;
            tmp[1] = c0*a + c3*b + c6*c - c7*d - c4*e - c1*f - c2*g - c5*h;
            tmp[2] = c0*a + c5*b - c6*c - c1*d - c4*e + c7*f + c2*g + c3*h;
            tmp[3] = c0*a + c7*b - c2*c - c5*d + c4*e + c3*f - c6*g - c1*h;
            tmp[4] = c0*a - c7*b - c2*c + c5*d + c4*e - c3*f - c6*g + c1*h;
            tmp[5] = c0*a - c5*b - c6*c + c1*d - c4*e - c7*f + c2*g - c3*h;
            tmp[6] = c0*a - c3*b + c6*c + c7*d - c4*e + c1*f - c2*g + c5*h;
            tmp[7] = c0*a - c1*b + c2*c - c3*d + c4*e - c5*f + c6*g - c7*h;
            */
            //Factored approx 30% faster than unfactored
            float sae,dae, sc2g, dc6g;
            sae = c0*a + c4*e;
            dae = c0*a - c4*e;
            sc2g = c2*c + c6*g;
            dc6g = c6*c - c2*g;
            tmp[0] = sae + c1*b + sc2g + c3*d + c5*f + c7*h;
            tmp[1] = dae + c3*b + dc6g - c7*d - c1*f - c5*h;
            tmp[2] = dae + c5*b - dc6g - c1*d + c7*f + c3*h;
            tmp[3] = sae + c7*b - sc2g - c5*d + c3*f - c1*h;
            tmp[4] = sae - c7*b - sc2g + c5*d - c3*f + c1*h;
            tmp[5] = dae - c5*b - dc6g + c1*d - c7*f - c3*h;
            tmp[6] = dae - c3*b + dc6g + c7*d + c1*f + c5*h;
            tmp[7] = sae - c1*b + sc2g - c3*d - c5*f - c7*h;
            

            img = tmp[n%8];
        
        ''',
        'idct_2d'
        ) (dct, w, h, img)
    return img

def dct_cupy_2d(img):
    dct_x = dct_cupy_2dx(img, bias=0)
    dct = dct_cupy_2dx(dct_x.T).T
    return dct

def idct_cupy_2d(dct):
    idct_y = idct_cupy_2dx(dct.T).T
    img = idct_cupy_2dx(idct_y)
    return img


def run_length_encode_block(dct_block, order=JPEC_ZZ_ENC):
    """ Run-length encodes the quantized DCT matrix """
    if len(set(order)) != len(order):
        raise Exception('Order contains duplicates!')
    if order.min() != 0 or order.max() != 63:
        raise Exception('Order contains values outside [0,63]')
    if order.dtype != np.int32:
        raise Exception('Order type is not np.int32')
    # Reorder
    reordered_raw = dct_block.flatten()[order]
    # Delta encode
    #reordered = np.zeros_like(reordered_raw)
    #reordered[0] = reordered_raw[0]
    #reordered[1:] = np.diff(reordered_raw)
    reordered = reordered_raw; # NO DELTA ENCODING
    # Early exit
    if not reordered.any():
        return [(0,0)] # Nothing to show, go right to EOB
    rv_pairs = []
    r = 0
    for x in reordered:
        if x != 0:
            rv_pairs += [(r, x)]
        elif r == 15:
            rv_pairs += [(15,0)]
            r = 0
        else:
            r += 1
    # Remove any tailing zeros (assumed after EOB)
    while rv_pairs[-1][1] == 0:
        rv_pairs.pop()
    rv_pairs += [(0,0)] # End of Block
    return rv_pairs   


def generate_huffman_table(values):
    syms,counts = np.unique(values, return_counts=1)
    #weights = 1.0*counts/counts.sum()
    heap = [[wt, [sym, '']] for sym,wt in zip(syms, counts)]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return dict(heappop(heap)[1:])


def jpec_cupy_scramble(img, dqt):
    
    dct = cuda.cupy.rint(dct_cupy_2d(img)).astype('f') # Some weirdness from not rounding these
    quant = quant_enc_cupy(dct, dqt)
    quant = quant_dec_cupy(quant, dqt)
    img_scrambled = cuda.cupy.rint(idct_cupy_2d(quant)).astype('f')
    #img_scrambled = idct_cupy_2d(quant)
    
    return img_scrambled

def _zvc_compressed_size(quant, bitwidth=8):
    # Total size is index, # zeros*bitwidth
    bitmask_size = 1 * quant.size     # Bits
    value_size = bitwidth * (quant != 0).sum()
    return value_size/8., bitmask_size/8.

def jpeg_cupy_compressed_size(img, dqt, order=JPEC_ZZ_ENC, force_signed=False, zvc=False):
    """ Returns compressed sizes (in Bytes) of encoded image and huffman table """
    nxy = np.array(img.shape)/8
    if (nxy.astype('i') != nxy.astype('f')).any():
        raise Exception('Image shape is not divisible by 8 {}'.format(img.shape))
    ny, nx = nxy.astype('i')
    
    xp = cuda.get_array_module(img)
    
    # Convert to signed integer
    signed = force_signed or img.min() < 0
    # Calculate signed transform to an int in 127,-128
    wbits,fbits = 3,5
    b = wbits + fbits     # Total bits
    upper = 2**b - 2**(b-1) - 1
    lower = -2**b + 2**(b-1)
    shift = 2.0**fbits
    bias = 0
    
    if signed:
        x = img*shift + bias
    else:
        x = img*shift - (img==0)*40 + bias
    xp.clip(x, lower, upper, out=x)
    xp.trunc(x, out=x)
    x = x.astype('f')
    
    # Run DCT and quantize
    dct = cuda.cupy.rint(dct_cupy_2d(x)).astype('f')
    quant = quant_enc_cupy(dct, dqt)
    
    # ZVC (if specified)
    if zvc:
        return _zvc_compressed_size(quant, bitwidth=8)
    
    # Everything from here on is on the CPU
    quant = cuda.to_cpu(quant)
    
    get_blk = lambda mat, m, n: mat[m*8:(m+1)*8, n*8:(n+1)*8]
    
    # Run-length encoding
    rle_frames = [[None for _ in range(nx)] for _ in range(ny)]
    for blk_x in range(nx):
        for blk_y in range(ny):
            #blk = quant[blk_y*8:(blk_y+1)*8, blk_x*8:(blk_x+1)*8]
            blk = get_blk(quant, blk_y, blk_x)
            rle = run_length_encode_block(blk, order=order)
            rle_frames[blk_y][blk_x] = rle
    
    
    # Get huffman symbol table
    rle_flat = sum(sum(rle_frames, []), [])
    ht = generate_huffman_table(list(zip(*rle_flat))[1]) # second element is value 
    
    # Applies huffman coding JPEG codes to a block
    def mapfn(rv):
        r,v = rv
        if r == 0 and v == 0:
            return (0,0,0,'') # EOB codeword
        if r == 15 and v == 0:
            return (r,v,0,'') # 15x0 codeword
        return (r,v, len(ht[v]), ht[v])
    
    encoded = []
    for blk_x in range(nx):
        for blk_y in range(ny):
            encoded += map(mapfn, rle_frames[blk_y][blk_x])
    
    
    # Encoded huffman table size, in bits
    # Each entry is roughly as follows:
    #    int[32] -> (nbits[4], bits[?])
    size_ht = sum(32 + 4 + len(val) for val in ht.values())
    
    # Entropy encoded run-length size, in bits
    size_rv = sum( 8+bits for r,v,bits,binary in encoded )
    
    # Return, in bytes
    return size_rv/8., size_ht/8.
    

class JPEGHelper(Helper):
    def __init__(self, dqt, bits=(3,5), dqt_name = None):
        
        dqt, dqt_name = _parse_dqt(dqt, dqt_name)
        
        self.dqt_name = dqt_name
        self.dqt = dqt.astype('f').flatten()
        self.bits = bits
        
        self.name = 'jpeg'
        self.detail = 'jpeg-{}-{}.{}'.format(self.dqt_name, self.bits[0], bits[1])
        
        self.is_setup = False
    
    def scramble_setup(self):
        self.touched = set()
        
    def print_settings(self, printfn=print):
        printfn('#   bits: {}'.format(self.bits))
        printfn('#   shift: x{}'.format(2.0**self.bits[1]))
        printfn('#   dqt: {}'.format(self.dqt_name))
        printfn('#     value:')
        for line in self.dqt.reshape((8,8)):
            printfn('#     ' + ','.join('%4d'%v for v in line) +',')
        
    def scramble(self,rank, function, var, force_signed=False):
        ''' Introduce error to the variable '''
        
        if var.data is None:
            raise Exception('None encountered in {}'.format(var.name))
            
        x  = var.data
        xp = cuda.get_array_module(x)
        num_imgs, num_chan, w,h = x.shape
        
        if not self.is_setup:
            self.dqt = cuda.to_gpu(self.dqt, cuda.get_device_from_array(x))
            self.is_setup = True
            
        if w != h or w<8:
            raise Exception('Shape for {} is {} (too small!) at {}'.format(var.name, x.shape, var.name))
        
        signed = force_signed or x.min() < 0
        
        # Calculate signed transform to an int in 127,-128
        wbits,fbits = self.bits   # Whole and fractional bits
        b = wbits + fbits     # Total bits
        
        # We add a bias for signed values
        # The range becomes 
        #  upper bound    2^b - 2^(b-1) - 1
        #  lower bound   -2^b + 2^(b-1)
        upper = 2**b - 2**(b-1) - 1
        lower = -2**b + 2**(b-1)
        shift = 2.0**fbits
        bias = 0
        
        # Equivalent to below but with in-place operations
        # x = (x*shifts + bias).clip(upper, lower).trunc() 
        x *= shift
        if not signed:
            x -= (x == 0) * 40
        if bias != 0:
            x += bias
        xp.clip(x, lower, upper, out=x)
        #xp.trunc(x, out=x)
        xp.rint(x, out=x)
        
        ## Reshape so all channels are one image
        images = x.view()
        images = x.reshape((num_imgs*num_chan*h, w))
        
        ## Pad the image if we don't have some multiple of 8
        h2 = num_imgs*num_chan*h
        padding = []
        if w%8 or h2%8:
            pad_w = (8 - w%8)%8
            pad_h = (8 - h2%8)%8
            padding = [pad_h, pad_w]
            images = xp.pad(images, [(pad_h,0), (pad_w,0)], 'constant', constant_values=0)
            
        
        if images.dtype != np.float32:
            raise Exception('images dtype is actually {}'.format(type(images)))
        
        # Do JPEG compression/decompression
        images_scram = jpec_cupy_scramble(images, self.dqt)
        
        # Undo padding
        if padding:
            images_scram = images_scram[padding[0]:, padding[1]:]
            
        # Undo channel grouping
        images = images_scram.view()
        images.shape = (num_imgs, num_chan, h, w)
        if images.dtype != np.float32:
            raise Exception('images dtype is actually {}'.format(type(images)))
        xp.trunc(images, out=images)
        
        # Undo shifting
        #images_final = (images-bias) / shifts # This operation makes a copy of the data
        if bias != 0:
            images -= bias
        images *= (1./shift)
        
        # Zero out, and add our images back in (this updates the original reference)
        x *= 0
        x += images
        
        if not signed:
            x *= (x>0).astype('f')
        
        # Original Method, adjust manually
        # These values were determined empirically by 
        # calculating the mean of the error for the first training iteration
        # adjust = (9.5199E-4)*x_range if not signed else (-4.8515E-5)*x_range
        # adjust = xp.mean(x - x_old)
        # x -= adjust
        #
        
        
        return True

class FastJPEGHelper(Helper):
    def __init__(self, dqt, bits=(3,5), dqt_name=None, cache_sign=True):
        
        dqt, dqt_name = _parse_dqt(dqt, dqt_name)
        
        self.dqt_name = dqt_name
        self.dqt = dqt.astype('f').flatten()
        self.bits = bits
        
        self.name = 'fjpeg'
        self.detail = 'fjpeg-{}-{}.{}'.format(self.dqt_name, self.bits[0], bits[1])
        
        self.is_setup = False
        
        
        self.cache_sign = cache_sign
        self.cache = None
    
    def scramble_setup(self):
        if self.cache_sign and self.cache is None:
            self.cache = {} # Starting the cache
        
    def print_settings(self, printfn=print):
        printfn('#   bits: {}'.format(self.bits))
        printfn('#   shift: x{}'.format(2.0**self.bits[1]))
        printfn('#   dqt: {}'.format(self.dqt_name))
        printfn('#     value:')
        for line in self.dqt.reshape((8,8)):
            printfn('#     ' + ','.join('%4d'%v for v in line) +',')
        
    def scramble(self,rank, function, var, force_signed=False):
        ''' Introduce error to the variable '''
        
        if var.data is None:
            raise Exception('None encountered in {}'.format(var.name))
            
        x  = var.data
        xp = cuda.get_array_module(x)
        num_imgs, num_chan, w,h = x.shape
        
        if not self.is_setup:
            self.dqt = cuda.to_gpu(self.dqt, cuda.get_device_from_array(x))
            self.is_setup = True
            
        if w != h or w<8:
            raise Exception('Shape for {} is {} (too small!) at {}'.format(var.name, x.shape, var.name))
        
        if force_signed:
            signed = True
        elif self.cache is not None:
            key = (rank, var.name, function.label if function is not None else None)
            if key in self.cache:
                signed = self.cache[key]
            else:
                signed = int(x.min() < 0)
                self.cache[key] = signed
        else:
            signed = int(x.min() < 0)
        
        
        #bias = 0 if signed else -128
        bias = 0
        zero_adjust = -40 if signed else 0
        
        # Use cupy kernel to compress to int
        cuda.elementwise(
            'float32 bias, float32 adj',
            'float32 x',
            ''' 
                float sh = ldexp(x, 5);
                float za = (sh == 0) ? sh + adj + bias : sh + bias; 
                int y = __float2int_rn(za);
                x = (float) ((y <= -128) ? -128 : 
                             (y >=  127) ?  127 : y ); // x = clip(y, -128, 127)
            ''', 
            'jpeg_to_fix35') (bias, zero_adjust, x)
        
        
        ## Reshape so all channels are one image
        images = x.view()
        images = x.reshape((num_imgs*num_chan*h, w))
        
        ## Pad the image if we don't have some multiple of 8
        h2 = num_imgs*num_chan*h
        padding = []
        if w%8 or h2%8:
            pad_w = (8 - w%8)%8
            pad_h = (8 - h2%8)%8
            padding = [pad_h, pad_w]
            images = xp.pad(images, [(pad_h,0), (pad_w,0)], 'constant', constant_values=0)
            
        
        if images.dtype != np.float32:
            raise Exception('images dtype is actually {}'.format(type(images)))
        
        # Do JPEG compression/decompression
        images_scram = jpec_cupy_scramble(images, self.dqt)
        
        # Undo padding
        if padding:
            images_scram = images_scram[padding[0]:, padding[1]:]
            
        # Undo channel grouping
        images = images_scram.view()
        images.shape = (num_imgs, num_chan, h, w)
        if images.dtype != np.float32:
            raise Exception('images dtype is actually {}'.format(type(images)))
        xp.trunc(images, out=images)
        
        # Undo fix35 transform
        cuda.elementwise(
            'float32 y, float32 bias',
            'float32 x',
            ''' 
                x = ldexp(y - bias, -5);   // x = (y-b) / 32
            ''', 
            'jpeg_from_fix35') (images, bias, x)
        
        if not signed:
            x *= (x>0).astype('f')
        
        # Original Method, adjust manually
        # These values were determined empirically by 
        # calculating the mean of the error for the first training iteration
        # adjust = (9.5199E-4)*x_range if not signed else (-4.8515E-5)*x_range
        # adjust = xp.mean(x - x_old)
        # x -= adjust
        #
        
        
        return True
    
def _parse_dqt(dqt_arg, name=None):
    """ Returns the parsed dqt, it's formatted name, and a string representation to be printed """
        
    def get_lines(dqt):
        dqt = dqt.reshape((8,8))
        return [','.join('%4d'%v for v in line) for line in dqt]
    
    dqt = None
    dqt_name = None
    
    if isinstance(dqt_arg, np.ndarray):
        dqt = dqt_arg.copy()
        dqt_name = 'const{},{},...{},{}'.format(dqt[0], dqt[1], dqt[-2], dqt[-1])
        
    elif isinstance(dqt_arg, str):
        if dqt_arg in QUANT_TABLES:
            dqt = QUANT_TABLES[dqt_arg]
            dqt_name = dqt_arg
            
        elif dqt_arg.startswith('band'):
            dqt = banded_dqt(dqt_arg)
            dqt_name = dqt_arg
            
        elif dqt_arg.startswith('fill'):
            dqt_str = dqt_arg[4:]
            if not (set(dqt_str) - set('1234567890,') == set()):
                argparse.ArgumentTypeError('Invalid characters in DQT "{}"'.format(dqt_arg))
            
            dqt = np.ones((64,)) * int(dqt_str)
            dqt_name = 'fill{}'.format(int(dqt[0]))
            
        elif dqt_arg.startswith('const'):
            dqt_str = dqt_arg[5:]
            if not (set(dqt_str) - set('1234567890,') == set()):
                argparse.ArgumentTypeError('Invalid characters in DQT "{}"'.format(dqt_arg))
            dqt = np.array([int(v) for v in dqt_str.split(',')])
            dqt_name = 'const{},{},...{},{}'.format(dqt[0], dqt[1], dqt[-2], dqt[-1])
            
        elif dqt_arg.startswith('jpeg'):
            quality = int(dqt_arg[4:])
            dqt = jpeg_dqt(quality)
            dqt_name = dqt_arg
            
    
    if dqt is None:
        raise argparse.ArgumentTypeError('Unable to parse DQT "{}" not found!'.format(dqt_arg))
        
    if dqt.shape != (8,8) and dqt.shape != (64,):
        raise argparse.ArgumentTypeError('DQT shape {} is invalid, must be (8,8) or (64,)'.format(dqt.shape))
            
        
    dqt_name = name or dqt_name
    if dqt_name is None:
        dqt_name = ','.join(str(int(v)) for v in dqt.diagonal())
    
    return dqt, dqt_name
        
    
JPEGHelper = FastJPEGHelper

if __name__ == '__main__':
    import numpy as np
    import time
    import chainer
    dev = 0
    for i in range(30):
        helpers = {}
        helpers['Normal'] = JPEGHelper('jpeg80')
        helpers['Fast'] = FastJPEGHelper('jpeg80')
        output = {}
        data = cuda.to_gpu(8*np.random.randn(128,256,8,8).astype('f'), device=dev)
        if np.random.rand() > 0.5:
            data = abs(data)
        for name, h in helpers.items():
            h.scramble_setup()
            x = data.copy()
            v = chainer.Variable(x)
            v.name = 'v%d'%i
            st = time.time()
            h.scramble(None, None, v)
            et = time.time()
            output[name] = x.copy()
            print('{:10}\t{:10} seconds'.format(name, et - st))
        for name, o in output.items():
            if name is 'Normal':
                continue
            diff = o - output['Normal']
            print('{} diff to Normal: {}'.format(name, abs(diff).max()))
        print('')
        
    print(data[0,0,0,:5])
    print(output['Normal'][0,0,0,:5])
    print(output['Fast'][0,0,0,:5])
