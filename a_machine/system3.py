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
test = col()
gamma = T.dscalar()

k_distance = k( observations.dimshuffle(0,'x',1,2), observations.dimshuffle('x',0,1,2), gamma )
test_k_distance = k( observations.dimshuffle(0,'x',1,2), test.dimshuffle('x',0,1,2), gamma )

distance = function( [observations, gamma], k_distance )
test_distance = function( [observations, test, gamma], test_k_distance)

class model:
  def __init__(self, gamma, sequence_length, classes):
    self.sequence_length = sequence_length
    self.gamma = gamma
    self.sequences = None
    self.k = None
    self.SVMs = None
    self.classes = classes
    
  # data: [observation][dimension]
  def train(self, data, labels):
    print "--> Training on %s %s-element subsequences, gamma=%s" % (data.shape[0] - self.sequence_length, self.sequence_length, self.gamma)

    #sequences = data.reshape(data.shape[0], 1, data.shape[1])
    #for i in np.arange( 1, self.sequence_length ):   
    #  sequences = np.hstack([ sequences, np.roll(data, -i, 0).reshape(data.shape[0], 1, data.shape[1]) ])
    #self.sequences = np.array( sequences )[:-self.sequence_length,:,:].astype('float32')
    self.sequences = self.make_sequences(data)
    
    # sequences: [sequence][observation][dimension]
    
    self.labels = labels[self.sequence_length:]
    
    self.k = distance(self.sequences, self.gamma )
    #print "--> %s non-null kernel distances" % ( (self.k > .00001).sum() )
    # NOTE:  consider masking the kernel matrix with a small value
    # We're computing a range of kernel widths, loosing some fidelity at the tight
    # end shouldn't be a huge problem
    
    # construct an SVM for each dimension of the observation data
    SVMs = []
    for i in range(data.shape[1]):
      SVM = svm.NuSVC( nu = .2, kernel='precomputed', probability=True)
      SVM.fit(self.k, self.labels[:,i])
      SVMs.append(SVM)
    self.SVMs = SVMs

  # data: [observation][dimension]      
  def predict(self, data):
    print "--> Predicting on %s sequences, gamma=%s" % (data.shape[0] - self.sequence_length, self.gamma)
    
    # [sequence][point][dimension]
    points = self.make_sequences(data)
    
    # [train_i][test_j]
    k = test_distance(self.sequences, points, self.gamma ).T
    
    # predictions: [test_sequence][dimension][class]
    predictions = np.array([]).reshape(points.shape[0], 0, self.classes.shape[0])
    
    for SVM in self.SVMs:
      # NOTE: this is returning |classes|+1 results ???
      prediction = np.expand_dims( SVM.predict_proba(k), 1)
      predictions = np.hstack( [predictions, prediction] )
    return predictions
    
  def make_sequences(self, data):
    sequences = data.reshape(data.shape[0], 1, data.shape[1])
    for i in np.arange( 1, self.sequence_length):
      sequences = np.hstack([ sequences, np.roll(data, -i, 0).reshape(data.shape[0], 1, data.shape[1]) ])
    
    return np.array( sequences )[:-self.sequence_length,:,:].astype('float32')

    
    