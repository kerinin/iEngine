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

def cs_divergence(P,Q, gamma):
  # If P is a single distribution, reshape it to [distribution(1)][point][dimension]
  if P.ndim == 2:
    P = P.dimshuffle('x',0,1,2)
  
  return cs_divergence_function(P,Q, gamma)


		
