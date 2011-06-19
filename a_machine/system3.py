#! /usr/bin/env python
import sys
from math import *
from datetime import *
import numpy as np
import scikits.statsmodels.api as sm
import scipy as sp

import matplotlib.pyplot as plt
import matplotlib.cm as cm

#import cs_divergence, parzen_probability
from scikits.learn import svm

from gpu_funcs import kernel_matrix
from svms import NuSVR
    
class model:
  def __init__(self, gamma_samples=1000, gamma_quantile=100, sequence_length=2):
    self.gamma_samples = gamma_samples
    self.gamma_quantile = gamma_quantile
    self.sequence_length = sequence_length
    self.gammas = None
    self.sequences = None
    self.k = None
    self.SVMs = None
    
  # data: [observation][dimension]
  def train(self, data, samples):
    self.gammas = self.determine_gammas_from(data)
    
    sequences = self.make_sequences(data)
    labels = data[self.sequence_length:,:].astype('float32')
    
    # Randomly sample |samples| sequences
    full = np.hstack([sequences, np.expand_dims(labels,1)])
    np.random.shuffle(full)
    self.sequences = full[:samples,:-1,:]
    self.labels = full[:samples,-1,:]

    #print full[0,:,0]
    #print "%s -> %s" % (str(self.sequences[0,:,0]), self.labels[0,0])
    #print full[1,:,0]
    #print "%s -> %s" % (str(self.sequences[1,:,0]), self.labels[1,0])

    # [gamma][dimension]
    self.SVMs = []
    for gamma in self.gammas:
      print "Computing kernel with gamma=%s" % gamma
      kk = kernel_matrix(self.sequences, self.sequences, gamma)
      
      g_SVMs = []
      for dimension in range(data.shape[1]):
        l = self.labels[:,dimension]
        
        # NOTE: this is where you would branch for nu/C
        hyp = NuSVR(nu=.5)
        hyp.train(kk,l)
        g_SVMs.append(hyp)
        
        print "--> SVM Trained: %s percent SV's, risk=%s" % ( hyp.SV_percent, hyp.risk ) 
      self.SVMs.append(g_SVMs)


  # data: [observation][dimension]      
  def predict(self, data):
    
    # [sequence][point][dimension]
    points = self.make_sequences(data)
        
    # [test_sequence][gamma][dimension]
    predictions = np.array([]).reshape(points.shape[0], 0, data.shape[1])
    risks = np.array([]).reshape(1, 0, data.shape[1])
    for i in range( len(self.SVMs) ):
      gamma = self.gammas[i]
      g_SVMs = self.SVMs[i]
      
      # [train_i][test_j]
      kk = kernel_matrix(self.sequences, points, gamma).T
      print "Computed kernel with gamma=%s, %s non-null entries" % (gamma, (kk > .00001).sum())
    
      g_predictions = np.array([]).reshape(points.shape[0], 0)
      g_risk = np.array([]).reshape(1, 0)
      for dimension in range( len(g_SVMs) ):
        SVM = g_SVMs[dimension]
                
        # [test][dimension]
        prediction = np.expand_dims( SVM.predict(kk), 1)
        #print prediction
        
        # Normalize by risk
        #prediction = prediction * SVM.SV_loss
        #risk += SVM.risk
        
        g_predictions = np.hstack( [g_predictions, prediction] )
        g_risk = np.hstack( [ g_risk, np.array(SVM.risk).reshape(1,1) ])
        
      predictions = np.hstack( [predictions, np.expand_dims( g_predictions, 1) ])
      risks = np.hstack( [risks, np.expand_dims( g_risk, 1 ) ])
    
    #print data[:self.sequence_length+4,0]
    #print "%s -> %s" % (str(points[0,:,0]), predictions[0,0,0])
    #print "%s -> %s" % (str(points[1,:,0]), predictions[1,0,0])
      
    # For now, just average them
    #return predictions.sum(1) / len(self.gammas)
    return predictions, risks
    
  def make_sequences(self, data):
    sequences = data.reshape(data.shape[0], 1, data.shape[1])
    for i in np.arange( 1, self.sequence_length):
      sequences = np.hstack([ sequences, np.roll(data, -i, 0).reshape(data.shape[0], 1, data.shape[1]) ])
    
    return np.array( sequences )[:-self.sequence_length,:,:].astype('float32')

  def determine_gammas_from(self, data):
    g_samples = data.copy()
    np.random.shuffle(g_samples)
    g_samples = g_samples[:self.gamma_samples]
    g_diff = np.abs( g_samples.reshape(g_samples.shape[0],1,g_samples.shape[1]) - g_samples.reshape(1,g_samples.shape[0],g_samples.shape[1]) )
    g_diff = g_diff.reshape(g_samples.shape[1]*g_samples.shape[0]**2)
    g_percentiles = np.arange(self.gamma_quantile / 2,100,self.gamma_quantile).astype('float')
    
    gammas = []
    for i in g_percentiles:
      gammas.append( sp.stats.stats.scoreatpercentile(g_diff, i) )   
    
    return np.array(gammas)
    
