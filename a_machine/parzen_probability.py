#! /usr/bin/env python
from math import *
import numpy as np

import theano.tensor as T
from theano import function

def k(X,Y,gamma):
  return ( 1 / T.pow( 
    2 * pi * T.pow(gamma,2),
  Y.shape[-1]/2) ) * T.exp( 
    -T.pow(X-Y, 2) / (2 * T.pow(gamma, 2) ) 
  ) 
  
col = T.TensorType('float32', [False, False,False])
row = T.TensorType('float32', [False,False])

observations = col('parzen observations')     # [distribution][observation][dimension]
test_points = row('parzen test points')       # [test point][dimension]
gamma = T.fscalar('gamma')                    # value

probabilities = T.sum( 
  T.prod( 
    k(observations.dimshuffle(0,1,'x',2), test_points.dimshuffle('x','x',0,1), gamma),  # => [distribution][observation][test_point][dimension
  3), # => [distribution][observation][test_point]
1) / observations.shape[1]  # => [distribution][test point]
  
parzen_function = function( [observations, test_points, gamma], probabilities )

# Calculate the estimated probability of a set of test_points given a set of observations
def from_one(observations, test_points, gamma):
  if observations.ndim != 2:
    raise ValueError, 'observations must be a 2-dimensional array in the form [sample point][dimension], not shape %s' % str(observations.shape)
  elif test_points.ndim != 2:
    raise ValueError, 'test_points must be a 2-dimensional arran in the form [test point][dimension], not shape %s' % str(test_points.shape)
    
  return parzen_function( observations.dimshuffle('x',0,1,2), test_points, gamma)
  
# Calculate the estimated probability of a set of test_points given multiple sets of observations
def from_many( observation_sets, test_points, gamma):
  if observation_sets.ndim != 3:
    raise ValueError, 'observation sets must be a 3-dimensional array in the form [set][sample point][dimension], not shape %s' % str(observation_sets.shape)
  elif test_points.ndim != 2:
    raise ValueError, 'test_points must be a 2-dimensional arran in the form [test point][dimension], not shape %s' % str(test_points.shape)
    
  return parzen_function( observation_sets, test_points, gamma)
