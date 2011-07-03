#! /usr/bin/env python
import sys
from math import *
from datetime import *
import numpy as np
import scikits.statsmodels.api as sm
import scipy as sp

import matplotlib.pyplot as plt
import matplotlib.cm as cm


from gpu_funcs import kernel_matrix
from cvxopt import matrix
from cvxopt.solvers import lp

class model:
  def __init__(self, dimension, gamma_samples=1000, gamma_quantile=100):
    self.dimension = dimension
    self.gamma_samples = gamma_samples
    self.gamma_quantile = gamma_quantile
    self.gammas = None
    self.sequences = None
    
    self.active_slices = None
    self.all_slices = None
    
  # data: [observation][dimension]
  def train(self, data, slices=[[0,[0]]]):
    #self.gammas = self.determine_gammas_from(data)
    self.gammas = [.5,]
    print "Gammas determined: %s" % str(self.gammas)
    
    # [gamma][sequence offset][dimension]
    #self.active_slices = np.mgrid[0:1,0:data.shape[1]].T.reshape(data.shape[1],2).tolist()
    # Make a single slice consisting of the 1st sequence element and all 3 dimensions
    #self.active_slices = [ [0,[0,1]], [0,[0]] ]
    self.active_slices = slices
    
    # Working with 1 sequence element for now
    sequences = data[:-1].astype('float32').reshape(data.shape[0]-1,1)
    labels = data[1:].astype('float32').reshape(data.shape[0]-1,1)

    self.sequences = sequences
    self.labels = labels
    l = self.sequences.shape[0]
    
    print "Calculating kernel matrix"
    kk = kernel_matrix(self.sequences.reshape(l,1,1), self.sequences.reshape(l,1,1), self.gammas[-1])
    
    print "Constructing constraints"
    # column
    c = np.hstack( [
      np.zeros(l),
      [0,1]
    ]) 
    
    #[constraint][variable]
    A_A = np.hstack( [
      kk.sum(0) / l,
      [0,0]
    ] )
    #[constraint]
    b_A = np.ones(1)
    
    A_G = np.hstack( [
      np.zeros(l),
      [1,-1]
    ] )
    b_G = np.zeros(1)


    G_B = np.hstack( [
      self.labels.flatten() * kk.sum(0),
      [0,-1]
    ] )
    h_B = np.array( [self.labels.sum() ] )
    
    G_C = np.hstack( [
      -self.labels.flatten() * kk.sum(0),
      [-1,0]
    ] )
    h_C = np.array( [ -self.labels.sum() ] )
    
    # [variable][variable]
    G_D = np.hstack( [
      -np.identity(l),
      np.zeros((l,2))
    ])
    h_D = np.zeros(l)
    
    G_E = np.hstack( [
      np.zeros(l),
      [-1,0]
    ])
    h_E = np.zeros(1)

    G_F = np.hstack( [
      np.zeros(l),
      [0,-1]
    ])
    h_F = np.zeros(1)

    G = np.vstack([G_B,G_C,G_D,G_E,G_F])
    #G = np.vstack([G_B,G_C,G_D]).astype('float')
    h = np.hstack([h_B,h_C,h_D,h_E,h_F])
    #h = np.hstack([h_B,h_C,h_D]).astype('float')
    
    print G
    print h
    #A = np.expand_dims( A_G, 0) 
    A = np.vstack([A_A,A_G])
    #b = b_G 
    b = np.vstack([b_A,b_G])
    
    #print G.shape
    #print h.shape
    print A.shape
    print b.shape
    
    print "Solving"
    solution = lp( matrix(c), matrix(G), matrix(h), matrix(A), matrix(b) )
    
    print "Handling Solution"
    if solution['status'] == 'optimal':
      X = np.array( solution['x'][:-2] )
      R_emp = np.array( solution['x'][-1] )
      
      self.SV_mask = ( X < 0 )
      self.beta = np.ma.compress_rows( np.ma.array( X, mask = self.SV_mask ) ).astype('float32')
      self.SVx = np.ma.compress_rows( np.ma.array( sequences, mask = np.repeat( self.SV_mask, sequences.shape[1], 1) ) ).astype('float32')
      self.SVy = np.ma.compress_rows( np.ma.array( labels.reshape(labels.shape[0],1), mask = self.SV_mask ) ).astype('float32')
      self.nSV = self.beta.shape[0]
    
      #print self.SVx.shape
      #print self.SVy.shape
      #print self.nSV
      #print self.SV_mask
      print solution['x']
    print "--> SVM Trained: %s SV's of %s, Risk=%s" % ( self.nSV, self.SV_mask.shape[0], R_emp/X.shape[0] ) 


  # data: [observation][dimension]      
  def predict(self, data):
    
    # [sequence][point][dimension]
    #points = self.make_sequences(data)
    points = data.astype('float32')
    
    
    #print points.shape
    #print self.sequences.shape
    
    #print points.shape
    #print self.SVx.reshape(self.nSV,1,1).shape
    
    kk = kernel_matrix( points, self.SVx.reshape(self.nSV,1,1), self.gammas[-1] )
    
    #print kk.shape
    #print self.SVy.shape
    #print self.beta.shape
    
    prediction = (self.SVy.T * self.beta.T * kk ).sum(1)
    
    #print prediction.shape
    
    return prediction

  def random_sample(self, sequences, labels, samples):
    full = np.hstack([sequences, np.expand_dims(labels,1)])
    np.random.shuffle(full)
    self.sequences = full[:samples,:-1,:]
    self.labels = full[:samples,-1,self.dimension] 
    (full[:samples,:-1,:], full[:samples,-1,self.dimension] )
    
    
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
        kk.append( kernel_matrix(subset_X, subset_Y, gamma) )

    
    # Construct the sparse block diagonal of the kernel matrices and extend the labels to match
    # return bdiag(kk, format='csr')
    
    if len(kk) > 1:
      row = np.hstack(kk)
      
      zeros = np.zeros( (kk[0].shape[0] * (len(kk)-1), kk[0].shape[1] * (len(kk)-1)) )
      k_column = np.vstack( kk[1:] )

      base = sp.sparse.csr_matrix( np.hstack( 
          [ k_column, zeros ] 
      ) )
      KK = sp.sparse.vstack([row,base]).todense()

      return KK
    else:
      return kk[0]
    
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
    
