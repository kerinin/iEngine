#! /usr/bin/env python
from math import *
import numpy as np

import theano.tensor as T
from theano import function

# KL_divergence(P,Q) = sum_i P(i) log ( P(i) / Q(i) )
# In this case, we'll use the points in 'other' as i
#activation = - T.sum( parzen_function(sequences, other) * T.log( parzen_function(sequences,other) / parzen_function(other,other) ), 1 )
#activation_function = function( [sequences,other], activation )

col = T.TensorType('float32', [False, False])
row = T.TensorType('float32', [True, False])

p_values = col('P(i)')
q_values = row('Q(i)')

#divergence = - T.sum( parzen(P, test_points) * T.log( parzen(P, test_points) / parzen( Q, test_points ) ), 1 )

divergence = -T.sum( p_values * T.log( p_values / q_values ), 1 )
kl_divergence_function = function( [p_values, q_values], divergence)

def kl_divergence(P,Q,test_points):
  p_values = parzen_function(P, test_points).astype('float32')
  q_values = parzen_function(Q, test_points).astype('float32')
  
  return kl_divergence_function(p_values, q_values)
  
