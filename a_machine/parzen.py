#! /usr/bin/env python
from math import *
import numpy as np

import theano.tensor as T
from theano import function

def k_parzen(X,Y,gamma):
  return T.exp( -gamma * T.pow(X-Y,2) )
  
col = T.TensorType('float32', [False, False,False])
row = T.TensorType('float32', [False,False])

observations = col('parzen observations')     # [distribution][observation][dimension]
test_points = row('parzen test points')       # [test point][dimension]
gamma = T.fscalar('gamma')                    # value

probabilities = T.sum( 
  T.prod( 
    k_parzen(observations.dimshuffle(0,1,'x',2), test_points.dimshuffle('x','x',0,1), gamma),  # => [distribution][observation][test_point][dimension
  3), # => [distribution][observation][test_point]
1) / observations.shape[1]  # => [distribution][test point]
  
parzen_function = function( [observations, test_points, gamma], probabilities )


