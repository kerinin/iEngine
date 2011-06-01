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

# Calculate the estimated probability of a set of test_points given a set of observations
def from_one(observations, test_points, gamma):
  return parzen_function( observations.dimshuffle('x',0,1,2), test_points, gamma)
  
# Calculate the estimated probability of a set of test_points given multiple sets of observations
def from_many( observation_sets, test_points, gamma):
  return parzen_function( observation_sets, test_points, gamma)
