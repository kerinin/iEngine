#! /usr/bin/env python
from math import *
import numpy as np

import theano.tensor as T
from theano import function


# Theano 

#def parzen(set,point):
#  return T.sum( k(set,point), 1) / set.shape[1]

gamma = 100

def k(X,Y):
  return T.prod(T.exp( -gamma * T.pow(X-Y,2) ), 3)
  
def parzen(observations, test_points):
  return T.sum( k(observations, test_points), 1) / observations.shape[1]

# matrix types for parzen estimation between sets of points and sets of sequences
col = T.TensorType('float32', [False, False,True,False])
row = T.TensorType('float32', [True, True,False,False])

observations = col('parzen observations')
test_points = row('parzen test points')
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
parzen_function = function( [observations, test_points], parzen(observations, test_points) )
	
	

# KL_divergence(P,Q) = sum_i P(i) log ( P(i) / Q(i) )
# In this case, we'll use the points in 'other' as i
#activation = - T.sum( parzen_function(sequences, other) * T.log( parzen_function(sequences,other) / parzen_function(other,other) ), 1 )
#activation_function = function( [sequences,other], activation )

col = T.TensorType('float32', [False, False])
row = T.TensorType('float32', [True, False])

p_values = col('P(i)')
q_values = row('Q(i)')

#divergence = - T.sum( parzen_function(P, test_points) * T.log( parzen_function(P, test_points) / parzen_function( Q, test_points ) ), 1 )
#divergence = - T.sum( parzen(P, test_points) * T.log( parzen(P, test_points) / parzen( Q, test_points ) ), 1 )

divergence = -T.sum( p_values * T.log( p_values / q_values ), 1 )
kl_divergence_function = function( [p_values, q_values], divergence)

def kl_divergence(P,Q,test_points):
  p_values = parzen_function(P, test_points).astype('float32')
  q_values = parzen_function(Q, test_points).astype('float32')
  
  return kl_divergence_function(p_values, q_values)
  

class model:
  def __init__(self):
    # create first layer
    pass
    
  def process(point):
    # pass point to first layer
    pass
      
class layer:
  def __init__(self, lower_level=None):
    # set lower level
    self.lower_level = lower_level
    self.upper_level = None
    self.sequences = []
    
    # set lower level's upper level
    if lower_level:
      lower_level.upper_level = self
    
    self.vectors = []

  def process(point):
    # append sequence to vectors
    self.sequences += (self.sequences[-1][1:] + [point])
    
    # calculate vector activation
    self.activation = activation_function(self.sequences, self.sequences[-1])

    # pass activation vector to next layer (possibly create next layer)
    if self.sequences.count == 1:
      layer(self)
    
    self.upper_level.process( self.sequences.map(activation) )

		
