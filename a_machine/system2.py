#! /usr/bin/env python
from math import *
from datetime import *
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cs_divergence, parzen_probability

import theano.tensor as T
from theano import function

def k(X,Y,gamma):
  return T.exp(-T.pow(X-Y,2)/(2*gamma**2))
  
col = T.TensorType('float32', [False, False, False])
row = T.TensorType('float32', [False, False])

data = col()
conditional = row()
gamma = T.fvector()

k_distance = k( data.dimshuffle(0,1,'x',2), conditional.dimshuffle('x','x',0,1), gamma.dimshuffle('x','x','x',0) )

distance = function( [data, conditional, gamma], k_distance )

class model:
  # assuming we'll get all the training data in one big string
  # IE, this isn't intended to be an online system
  def __init__(self, sequence_length, gamma):
    self.sequence_length = sequence_length
    self.gamma = gamma.astype('float32')
    
    # [data set][observation][dimension]
    self.data = None

  def process(self, new_data):
    if self.data == None:
      self.data = np.expand_dims(new_data, axis=0)
    else:
      if new_data.shape[0] != self.data.shape[1]:
        raise ValueError, 'new data must be the same size as previous chunks (%s, not %s)' % (self.data.shape[1], new_data.shape[0])
      
      self.data = np.vstack( [ self.data, np.expand_dims(new_data, axis=0) ] )
    
  def predict_from(self, sequence):
    if sequence.shape[0] != self.sequence_length:
      raise ValueError, 'conditional sequence must have length %s, not %s' % (self.sequence_length, sequence.shape[0])
    elif sequence.shape[1] != self.data.shape[2]:
      raise ValueError, 'conditional sequnce must contain %s-dimensional observations, not %s' % (self.data.shape[2], sequence.shape[1])
      
    #print self.data.shape
    #print sequence.shape
    #print self.gamma.shape
    
    # compute the kernel distance between the sequence and each observed point
    dd = distance( self.data.astype('float32'), sequence.astype('float32'), self.gamma )
    #print dd.shape  # => [training set][training point][sequence point][dimension]
    
    # append the target points
    dd = np.dstack( [dd, np.expand_dims(self.data, axis=2) ] )
    #print dd.shape  # => [training set][training point][sequence point + target][dimension]
    
    # shift the points to create sequences
    for i in np.arange(1, dd.shape[2]):
      dd[:,:,:-i,:] = dd[:,:,i:,:]
      
    # drop the partial sequences
    dd = dd[:,:dd.shape[1]-dd.shape[2],:,:]
    #print dd.shape
    
    # combine training sets
    dd = dd.reshape( dd.shape[0] * dd.shape[1], dd.shape[2], dd.shape[3] )
    
    # calculated weighted average
    d_cond = dd[:,:-1,:]
    d_pred = dd[:,-1,:]
    activation = d_cond.prod(2).prod(1).reshape(d_cond.shape[0],1)
    average = (d_pred * activation).sum(0) / d_pred.shape[0]
    #print average.shape
    print average
    #return
    
    return average
    
    