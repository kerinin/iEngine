#! /usr/bin/env python
from math import *
from datetime import *
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

#import cs_divergence, parzen_probability
from scikits.learn import svm

import theano.tensor as T
from theano import function


# [sequence_i][sequence_j][observation][dimension]
# (sqrt(2pi)sigma)^-d * exp( x^2 / -2sigma^2)
def k(X,Y,gamma):
  return T.prod( T.prod( T.exp(-T.pow(X-Y,2)*gamma), 3),  2)
  #return T.prod( T.prod( T.exp(-T.pow(X-Y,2)/(2*gamma**2)), 3),  2)
  return T.prod( T.prod( 
    T.pow(sqrt(gamma*2*pi), -X.shape[2]) * T.exp( T.pow(X-Y, 2) / (-2 * T.pow(gamma, 2))),
  3), 2)
  
col = T.TensorType('float32', [False, False, False])

X = col()
Y = col()
gamma = T.dscalar()

k_distance = k( X.dimshuffle(0,'x',1,2), Y.dimshuffle('x',0,1,2), gamma )

distance = function( [X, Y, gamma], k_distance )

def kernel_matrix(X,Y,gamma):
  # Procedure for splitting kernel into 2^n sub-problems
  # Avoids memory errors if array too large for GPU
  # Let's assume 64-bits/point (float + link)
  mem = 64 * X.shape[0] * Y.shape[0] * X.shape[1] * X.shape[2]
  #mem_available = 2e9 # 2e8 for laptop
  mem_available = 2e8
  n = int( ceil( sqrt( mem/mem_available ) ) )
  
  while True:
    l_n = ceil( float(X.shape[0]) / n )
    print "--> Trying with n=%s, %s sub-matrices of size %sx%s (mem estimate %s)" % (n, n**2, l_n, l_n, mem/mem_available)
    
    try:
      if n == 1:
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
          print "--> Row %s" % i
        return np.array(kk).reshape(X.shape[0], Y.shape[0])
    except MemoryError, RuntimeError:
      n += 1
    else:
      break
      
class model:
  def __init__(self, gamma, sequence_length):
    self.sequence_length = sequence_length
    self.gamma = gamma
    self.sequences = None
    self.k = None
    self.SVMs = None
    
  # data: [observation][dimension]
  def train(self, data):
    print "--> Training on %s %s-element subsequences, gamma=%s" % (data.shape[0] - self.sequence_length, self.sequence_length, self.gamma)

    self.sequences = self.make_sequences(data)
    self.labels = data[self.sequence_length:,:]
    
    self.labels = data[self.sequence_length:,:].astype('float32')

    self.k = kernel_matrix(self.sequences, self.sequences, self.gamma)
    
    print "--> %s non-null kernel distances" % ( (self.k > .00001).sum() )
    # NOTE:  consider masking the kernel matrix with a small value
    # We're computing a range of kernel widths, loosing some fidelity at the tight
    # end shouldn't be a huge problem
    
    # construct an SVM for each dimension of the observation data
    SVMs = []
    for i in range(data.shape[1]):
      SVM = svm.NuSVR( nu = .2, kernel='precomputed', probability=True, cache_size=1000)
      
      SVM.fit(self.k, self.labels[:,i])
      SVMs.append(SVM)
    self.SVMs = SVMs
    print "--> SVM's trained"

  # data: [observation][dimension]      
  def predict(self, data):
    print "--> Predicting on %s sequences, gamma=%s" % (data.shape[0] - self.sequence_length, self.gamma)
    
    # [sequence][point][dimension]
    points = self.make_sequences(data)
    
    # [train_i][test_j]
    k = kernel_matrix(self.sequences, points, self.gamma).T
    print "--> %s non-null kernel distances" % ( (k > .00001).sum() )
    
    
    # predictions: [test_sequence][dimension]
    predictions = np.array([]).reshape(points.shape[0], 0)
    
    for SVM in self.SVMs:
      # NOTE: this is returning |classes|+1 results ???
      prediction = np.expand_dims( SVM.predict(k), 1)
      predictions = np.hstack( [predictions, prediction] )
    return predictions
    
  def make_sequences(self, data):
    sequences = data.reshape(data.shape[0], 1, data.shape[1])
    for i in np.arange( 1, self.sequence_length):
      sequences = np.hstack([ sequences, np.roll(data, -i, 0).reshape(data.shape[0], 1, data.shape[1]) ])
    
    return np.array( sequences )[:-self.sequence_length,:,:].astype('float32')

    
    
