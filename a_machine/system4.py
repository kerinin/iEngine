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

import theano.tensor as T
from theano import function

col = T.TensorType('float32', [False, False, False])

X = col()
Y = col()
gamma = T.dscalar()

# [sequence_i][sequence_j][observation][dimension]
# (sqrt(2pi)sigma)^-d * exp( x^2 / -2sigma^2)
def k(X,Y,gamma):
  return T.prod( T.prod( 
    T.pow(gamma * T.sqrt(2*pi), -X.shape[2]) * T.exp( T.pow(X-Y,2) / (-2 * T.pow(gamma,2) ) ), 
  3),  2)
  


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
        sys.stdout.write('\n')
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
          #sys.stdout.write( " %s" % i )
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
    self.optimizer = svm.sparse.NuSVR(nu=nu, C=C, kernel='precomputed', cache_size=2000)
    self.SV_indices = None
    self.SV_kk = None
    self.SV_weights = None
    self.SV_loss = None
    self.SV_percent = None
    self.intercept = None
    self.risk = None

  def train(self, kk, values):
    self.optimizer.fit(kk, values)
    
    self.SV_indices = self.optimizer._support_indices
    #self.kk_SV = kk[self.SV_indices]
    self.SV_weights = self.optimizer._dual_coef_data
    self.loss = np.abs( self.optimizer.predict( kk ) - values )
    self.SV_percent = int( 100 * float( self.SV_indices.shape[0] ) / kk.shape[0]**2 )
    
    self.intercept = self.optimizer.intercept_
    self.risk = self.loss.sum() / kk.shape[0]
    
  def predict(self, kk):
    return self.optimizer.predict(kk)
    
    
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
    #self.active_slices = np.repeat( np.mgrid[ 0:1, 0:1, 0:data.shape[1] ].T.reshape(data.shape[1], 3), len(self.gammas), axis=0 )
    self.active_slices = np.mgrid[0:1,0:data.shape[1]].T.reshape(data.shape[1],2).tolist()
    print self.active_slices
    
    sequences = self.make_sequences(data)
    labels = data[self.sequence_length:,:].astype('float32')
    
    # Randomly sample |samples| sequences
    full = np.hstack([sequences, np.expand_dims(labels,1)])
    np.random.shuffle(full)
    self.sequences = full[:samples,:-1,:]
    self.labels = full[:samples,-1,self.dimension]
    
    KK = self.make_subsets(self.sequences, self.sequences)
    Labels = np.tile( self.labels, len(self.gammas) * len(self.active_slices) )
    
    print KK.shape
    print Labels.shape
    # Train that shist
    self.svm = SVM(nu=.5)
    self.svm.train(KK, Labels)
    print "--> SVM Trained: %s percent SV's, risk=%s" % ( self.svm.SV_percent, self.svm.risk ) 


  # data: [observation][dimension]      
  def predict(self, data):
    
    # [sequence][point][dimension]
    points = self.make_sequences(data)
    KK = self.make_subsets(points, self.sequences)
    
    raw = self.svm.predict(KK)
    print raw.shape
    #structured = raw.reshape(points.shape[0], len(self.gammas),len(self.active_slices))
    structured = raw.reshape(points.shape[0], len(self.active_slices), len(self.gammas))
    #structured = raw.reshape(len(self.gammas),len(self.active_slices), points.shape[0])
    #structured = raw.reshape(len(self.active_slices), len(self.gammas), points.shape[0])
    #structured = raw.reshape(len(self.active_slices), points.shape[0], len(self.gammas))
    #structured = raw.reshape( len(self.gammas), points.shape[0], len(self.active_slices))

    return structured / (len(self.gammas) * len(self.active_slices) )
    
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
        kk.append( kernel_matrix(subset_X, subset_Y, gamma) )

    
    # Construct the sparse block diagonal of the kernel matrices and extend the labels to match
    return bdiag(kk, format='csr')
    
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
    
