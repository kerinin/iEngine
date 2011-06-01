#! /usr/bin/env python
from math import *
import numpy as np

import theano.tensor as T
from theano import function


# Cauchy-Schwarz Divergence
# -log( Sum(K(P,Q)) / Sqrt( Sum(K(P,P)) * Sum(K(Q,Q)) ) )
# Where K(X,Y) is the kernel distance between the constituent points in two parzen estimators X, and Y
#
# This is a different formulation of the KL used above, but has the nice property of being
# bounded & symmetric, and has a nice paper explaining its derivation and application to parzen
# estimators.

def k_cs(X,Y, gamma):
  return T.exp( -gamma * T.pow(X-Y,2) )
  
# [distribution][point][dimension]
distributions = T.TensorType('float32', [False,False,False])
# [point][dimension]
distribution = T.TensorType('float32', [False,False])

P = distributions('P')
Q = distribution('Q')
gamma = T.fscalar('gamma')

divergence = -T.log( 
  T.sum(T.sum(
    T.prod(
      k_cs( P.dimshuffle(0,'x',1,2), Q.dimshuffle('x',0,'x',1), gamma ), # => [distribution][P_i][Q_j][dimension]
    3 ), # => [distribution][P_i][Q_j]
  1), 1) / # => [distribution]
  T.sqrt( 
    ( 
      T.sum(T.sum(
        T.prod(
          k_cs(P.dimshuffle(0,1,'x',2),P.dimshuffle(0,'x',1,2), gamma), # => [distribution][P_i][P_j][dimension]
        3 ), # => [distribution][P_i][P_j]
      1), 1) # => [distribution]
    ) *
    (
      T.sum(
        T.prod(
          k_cs(Q.dimshuffle(0,'x',1),Q.dimshuffle('x',0,1), gamma), # => [Q_i][Q_j][dimension]
        2 ), # => [Q_i][Q_j]
      ) # => []
    ).dimshuffle('x')
  ) # => [distribution]
)
cs_divergence_function = function( [P,Q,gamma], divergence)

# Calcualte the divergence of a distribution Q from a distribution P
def from_one(P,Q, gamma):
  if P.ndim != 2:
    raise ArgumentError, 'P must be a 2-dimensional array in the form [sample point][dimension]'
  elif Q.ndim != 2:
    raise ArgumentError, 'Q must be a 2-dimensional arran in the form [sample point][dimension]'
    
  return cs_divergence_function(P.dimshuffle('x',0,1,2), Q, gamma)
  
# Calculate the divergence of a distribution Q from a set of distributions PP
def from_many(PP,Q, gamma):
  if P.ndim != 3:
    raise ArgumentError, 'P must be a 3-dimensional array in the form [distribution][sample point][dimension]'
  elif Q.ndim != 2:
    raise ArgumentError, 'Q must be a 2-dimensional arran in the form [sample point][dimension]'
    
  return cs_divergence_function(PP,Q, gamma)
  

		
