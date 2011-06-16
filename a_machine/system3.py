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
  #return T.prod( T.prod( T.exp(-T.pow(X-Y,2)*gamma), 3),  2)
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
      
class SVM:

  # kk_sv = kk[clf.support_]
  # kk_sv = kk_sv[:,clf.support_]
  
  # Prediction works like this, bitches:
  # kk: [sample point][train point]
  # kk_sv: [sample point][SV]
  #kk_sv = kk[:,SVM.support_]
  #(kk_sv * SVM.dual_coef_).sum(1) + SVM.intercept_
  # R_emp = 1/ell sum_i L(|y_i - f(x)|)
  # L(x) = 0 (x<epsilon), x-epsilon (otherwise)
  # Looks like you'll need to determine the risk explicitly
  # The optimization problem doesn't actually solve for it directly
    
  def __init__(self, nu=.2, C=.5):
    self.nu = nu
    self.C = C
    self.optimizer = svm.NuSVR(nu=nu, C=C, kernel='precomputed', cache_size=2000)
    self.SV_indices = None
    self.SV_kk = None
    self.SV_weights = None
    self.SV_loss = None
    slef.intercept = None
    self.loss = None

  def train(self, kk, values):
    self.optimizer.fit(kk, values)
    
    self.SV_indices = self.optimizer.support_
    self.SV_kk = ( kk[self.SV_indices] )[:,self.SV_indices]
    self.SV_weights = self.optimizer.dual_coef_
    self.SV_loss = self.optimizer.predict( self.SV_kk ) - values
    
    self.intercept = self.optimizer.intercept_
    self.loss = self.SV_loss.sum()
    
    
class model:
  def __init__(self, gamma_samples=1000, gamma_quantile=100, sequence_length=2):
    self.gamma_samples = None
    self.gamma_quantile = None
    self.sequence_length = sequence_length
    self.gammas = None
    self.sequences = None
    self.k = None
    self.SVMs = None
    
  # data: [observation][dimension]
  def train(self, data):
    self.gammas = self.determine_gammas_from(data)
    
    self.sequences = self.make_sequences(data)
    self.labels = data[self.sequence_length:,:].astype('float32')

    self.SVMs = []
    for gamma in self.gammas:
      kk = kernel_matrix(self.sequences, self.sequences, gamma)
      
      for dimension in range(data.shape[1]):
        l = self.labels[:,dimension]
        
        # NOTE: this is where you would branch for nu/C
        hyp = SVM(nu=.2)
        hyp.train(kk,l)
        self.SVMs.append(hyp)


  # data: [observation][dimension]      
  def predict(self, data):
    
    # [sequence][point][dimension]
    points = self.make_sequences(data)
    
    # [test_sequence][gamma][dimension]
    predictions = np.array([]).reshape(points.shape[0], 0, 0)
    risk = 0
    for gamma in self.gammas:
      # [train_i][test_j]
      kk = kernel_matrix(self.sequences, points, gamma).T
      #print "--> %s non-null kernel distances" % ( (k > .00001).sum() )
    
    
      for SVM in self.SVMs:
        prediction = np.expand_dims( SVM.predict(kk), 1)
        
        # Normalize by risk
        prediction = prediction * SVM.SV_risk
        risk += SV.risk
        
        predictions = np.hstack( [predictions, prediction] )
        
    return predictions.sum(1) / risk
    
    
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
    
