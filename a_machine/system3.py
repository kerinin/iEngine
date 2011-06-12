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
def k(X,Y,gamma):
  return T.prod( T.prod( T.exp(-T.pow(X-Y,2)/(2*gamma**2)), 3),  2)
  
col = T.TensorType('float32', [False, False, False])

observations = col()
gamma = T.dscalar()

k_distance = k( observations.dimshuffle(0,'x',1,2), observations.dimshuffle('x',0,1,2), gamma )

distance = function( [observations, gamma], k_distance )

class model:
  def __init__(self, gamma, sequence_length):
    self.sequence_length = sequence_length
    self.gamma = gamma
    self.sequences = None
    self.k = None
    
  # data: [observation][dimension]
  def train(self, data, labels):
    print "--> Training on %s %s-element subsequences, gamma=%s" % (data.shape[0] - self.sequence_length, self.sequence_length, self.gamma)

    sequences = data.reshape(data.shape[0], 1, data.shape[1])
    for i in np.arange( 1, self.sequence_length ):   
      sequences = np.hstack([ sequences, np.roll(data, -i, 0).reshape(data.shape[0], 1, data.shape[1]) ])
    self.sequences = np.array( sequences )[:-self.sequence_length,:,:].astype('float32')
    
    # sequences: [sequence][observation][dimension]
    
    self.labels = labels[self.sequence_length:]
    
    self.k = distance(self.sequences, self.gamma )
    #print "--> %s non-null kernel distances" % ( (self.k > .00001).sum() )
    # NOTE:  consider masking the kernel matrix with a small value
    # We're computing a range of kernel widths, loosing some fidelity at the tight
    # end shouldn't be a huge problem
    
    SVMs = []
    for i in range(data.shape[1]):
      SVM = svm.NuSVC( nu = .1, kernel='precomputed', probability=True)
      SVM.fit(self.k, self.labels[:,i])
      SVMs.append(SVM)
    
    

    
    