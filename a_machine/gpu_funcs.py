#! /usr/bin/env python
import sys
from math import *
import numpy as np

import theano.tensor as T
from theano import function

col = T.TensorType('float32', [False, False, False])

X = col()
Y = col()
gamma = T.dscalar()

# [sequence_i][sequence_j][observation][dimension]
# (sqrt(2pi)sigma)^-d * exp( x^2 / -2sigma^2)
def k(X,Y,gamma):
  return T.prod( T.prod( 
    T.pow(gamma * T.sqrt(2*pi), -X.shape[2]) * T.exp( T.pow(X-Y,2) / (-2 * T.pow(gamma,2) ) ), 
  3),  2)
  
k_distance = k( X.dimshuffle(0,'x',1,2), Y.dimshuffle('x',0,1,2), gamma )

distance = function( [X, Y, gamma], k_distance )

def kernel_matrix(X,Y,gamma):
  # Procedure for splitting kernel into 2^n sub-problems
  # Avoids memory errors if array too large for GPU
  # Let's assume 64-bits/point (float + link)
  mem = 64 * X.shape[0] * Y.shape[0] * X.shape[1] * X.shape[2]
  mem_available = 2e9 # 2e8 for laptop
  #mem_available = 2e8
  n = int( ceil( sqrt( mem/mem_available ) ) )
  
  while True:
    l_n = ceil( float(X.shape[0]) / n )
    print "--> Trying with n=%s, %s sub-matrices of size %sx%s (mem estimate %s)" % (n, n**2, l_n, l_n, mem/mem_available)

    try:
      if n == 1:
        sys.stdout.write('\n')
        return distance(X, Y, gamma)
      else:         
        kk = np.array([]).reshape(0,Y.shape[0])
        
        for i in range(n):
          l_i1 = l_n * i
          l_i2 = X.shape[0] if l_n * (i+1) > X.shape[0] else l_n * (i+1)
          kk_i = np.array([]).reshape(l_i2-l_i1,0)
          
          for j in range(n):
            
            l_j1 = l_n * j
            l_j2 = Y.shape[0] if l_n * (j+1) > Y.shape[0] else l_n * (j+1)
            
            kk_j = distance( X[l_i1:l_i2,:,:], Y[l_j1:l_j2,:,:], gamma)
            
            kk_i = np.hstack([kk_i, kk_j])
            
          kk = np.vstack([ kk, kk_i ])
          #sys.stdout.write( " %s" % i )
        return np.array(kk).reshape(X.shape[0], Y.shape[0])
    except MemoryError, RuntimeError:
      n += 1
    else:
      break
    
