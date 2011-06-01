#! /usr/bin/env python
from math import *
import numpy as np

import theano.tensor as T
from theano import function

def k_parzen(X,Y,gamma):
  return T.prod(T.exp( -gamma * T.pow(X-Y,2) ), 3)
  
def parzen(observations, test_points, gamma):
  return T.sum( k_parzen(observations, test_points, gamma), 1) / observations.shape[1]

# matrix types for parzen estimation between sets of points and sets of sequences
col = T.TensorType('float32', [False, False,True,False])
row = T.TensorType('float32', [True, True,False,False])

observations = col('parzen observations')
test_points = row('parzen test points')
gamma = T.fscalar('gamma')
#probabilities = T.sum( k(observations, test_points), 1) / observations.shape[1]

# observations should be in format [sequence][observation][1][dimension]
# test_points should be in format [1][1][test_point][dimension]
# returns in the format [sequence][test_point]
# 
# example:
# 3 sequences defined by 10 points of 2 dimensions
# observations = np.random.rand(3,10,1,2)
# 5 test points of 2 dimensions
# test_points = np.random.rand(1,1,4,2)
# parzen_function(observations, test_points) => result[a][b] = the probability of test point (b) given sequence (a)
parzen_function = function( [observations, test_points, gamma], parzen(observations, test_points, gamma) )
