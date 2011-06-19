#! /usr/bin/env python
import sys
from math import *
from datetime import *
import numpy as np
import scikits.statsmodels.api as sm
import scipy as sp
from bdiag import *

import matplotlib.pyplot as plt
import matplotlib.cm as cm

#import cs_divergence, parzen_probability
from scikits.learn import svm

from gpu_funcs import kernel_matrix
from svms import sparseNuSVR    
    
class model:
  def __init__(self, dimension, gamma_samples=1000, gamma_quantile=100, sequence_length=2):
    self.dimension = dimension
    self.gamma_samples = gamma_samples
    self.gamma_quantile = gamma_quantile
    self.sequence_length = sequence_length
    self.gammas = None
    self.sequences = None
    self.k = None
    self.SVM = None
    
    # [gamma][sequences][dimensions]
    self.active_slices = None
    self.all_slices = None
    
  # data: [observation][dimension]
  def train(self, data, samples):
    self.gammas = self.determine_gammas_from(data)
    
    # [gamma][sequence offset][dimension]
    self.active_slices = np.mgrid[0:1,0:data.shape[1]].T.reshape(data.shape[1],2).tolist()
    
    sequences = self.make_sequences(data)
    labels = data[self.sequence_length:,:].astype('float32')
    
    # Randomly sample |samples| sequences
    #full = np.hstack([sequences, np.expand_dims(labels,1)])
    #np.random.shuffle(full)
    #self.sequences = full[:samples,:-1,:]
    #self.labels = full[:samples,-1,self.dimension]
    
    self.sequences = sequences[:samples]
    self.labels = labels[:samples,self.dimension]
    print self.sequences[:10]
    
    #KK = self.make_subsets(self.sequences, self.sequences)
    #Labels = np.hstack( [ self.labels, np.zeros( self.labels.shape[0] * ( len(self.gammas) * len(self.active_slices) - 1) ) ] )
    #weights = np.hstack( [ np.ones( self.labels.shape[0] ), np.zeros( self.labels.shape[0] * ( len(self.gammas) * len(self.active_slices) - 1 ) ) ] )
    
    print 'Training...'
    # Train that shist
    self.svm = sparseNuSVR(nu=.5)
    #self.svm.train(KK, Labels, sample_weight = weights)
    #self.svm.train(KK, self.labels)
    self.svm.train( kernel_matrix(self.sequences, self.sequences, self.gammas[-1]), self.labels)
    print "--> SVM Trained: %s percent SV's, risk=%s" % ( self.svm.SV_percent, self.svm.risk ) 


  # data: [observation][dimension]      
  def predict(self, data):
    
    # [sequence][point][dimension]
    points = self.make_sequences(data)
    print points[:10]
    #KK = self.make_subsets(points, self.sequences)
    KK = kernel_matrix( points, self.sequences, self.gammas[-1] )
    
    raw = self.svm.predict(KK)

    return raw
    
    
    
    
    
    
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
        
        g_predictions = np.hstack( [g_predictions, prediction] )
        g_risk = np.hstack( [ g_risk, np.array(SVM.risk).reshape(1,1) ])
        
      predictions = np.hstack( [predictions, np.expand_dims( g_predictions, 1) ])
      risks = np.hstack( [risks, np.expand_dims( g_risk, 1 ) ])

    return predictions, risks
    
  def make_sequences(self, data):
    sequences = data.reshape(data.shape[0], 1, data.shape[1])
    for i in np.arange( 1, self.sequence_length):
      sequences = np.hstack([ sequences, np.roll(data, -i, 0).reshape(data.shape[0], 1, data.shape[1]) ])
    
    return np.array( sequences )[:-self.sequence_length,:,:].astype('float32')

  def make_subsets(self, X, Y):
    kk = []
    for s in self.active_slices:
      if isinstance(s[1], int):
        subset_X = X[:,s[0],s[1]].reshape(X.shape[0], s[0]+1, 1 )
        subset_Y = Y[:,s[0],s[1]].reshape(Y.shape[0], s[0]+1, 1 )
      else:
        subset_X = X[:,s[0],s[1]].reshape(X.shape[0], s[0]+1, len(s[1]))
        subset_Y = Y[:,s[0],s[1]].reshape(Y.shape[0], s[0]+1, len(s[1]))

      for gamma in self.gammas:
        # NOTE:  returning to test on single matrix
        return kernel_matrix(subset_X,subset_Y,gamma)
        kk.append( kernel_matrix(subset_X, subset_Y, gamma) )

    
    # Construct the sparse block diagonal of the kernel matrices and extend the labels to match
    # return bdiag(kk, format='csr')
    
    row = np.hstack(kk)
    
    zeros = np.zeros( (kk[0].shape[0] * (len(kk)-1), kk[0].shape[1] * (len(kk)-1)) )
    k_column = np.vstack( kk[1:] )

    base = sp.sparse.csr_matrix( np.hstack( 
        [ k_column, zeros ] 
    ) )
    KK = sp.sparse.vstack([row,base])

    return KK
    
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
    
