#! /usr/bin/env python
import numpy as np
from scikits.learn import svm
    
class sparseNuSVR:
  def __init__(self, nu=.2, C=.5):
    self.nu = nu
    self.C = C
    self.optimizer = svm.NuSVR(nu=nu, C=C, kernel='precomputed', cache_size=2000)
    self.SV_indices = None
    self.SV_kk = None
    self.SV_weights = None
    self.SV_loss = None
    self.intercept = None
    self.risk = None

  def train(self, kk, values, sample_weight=None):
    if sample_weight == None:
      self.optimizer.fit(kk, values)
    else:
      self.optimizer.fit(kk, values, sample_weight=sample_weight)
    
    self.SV_indices = self.optimizer.support_
    #print self.SV_indices
    self.kk_SV = kk[self.SV_indices]
    self.SV_weights = self.optimizer.dual_coef_
    print self.SV_weights
    self.SV_loss = np.abs( self.optimizer.predict( self.kk_SV ) - values[self.SV_indices] )
    self.SV_percent = int( 100 * float( self.SV_indices.shape[0] ) / kk.shape[0] )
    
    self.intercept = self.optimizer.intercept_
    self.risk = self.SV_loss.sum() / kk.shape[0]
    
  def predict(self, kk):
    return self.optimizer.predict(kk)
    
    
class NuSVR:
  def __init__(self, nu=.2, C=.5):
    self.nu = nu
    self.C = C
    self.optimizer = svm.NuSVR(nu=nu, C=C, kernel='precomputed', cache_size=2000)
    self.SV_indices = None
    self.SV_kk = None
    self.SV_weights = None
    self.SV_loss = None
    self.intercept = None
    self.risk = None

  def train(self, kk, values, sample_weight=None):
    if sample_weight == None:
      self.optimizer.fit(kk, values)
    else:
      self.optimizer.fit(kk, values, sample_weight=sample_weight)
    
    self.SV_indices = self.optimizer.support_
    self.kk_SV = kk[self.SV_indices]
    self.SV_weights = self.optimizer.dual_coef_
    self.SV_loss = np.abs( self.optimizer.predict( self.kk_SV ) - values[self.SV_indices] )
    self.SV_percent = int( 100 * float( self.SV_indices.shape[0] ) / kk.shape[0] )
    
    self.intercept = self.optimizer.intercept_
    self.risk = self.SV_loss.sum() / kk.shape[0]
    
  def predict(self, kk):
    return self.optimizer.predict(kk)
