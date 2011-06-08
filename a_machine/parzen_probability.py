#! /usr/bin/env python
from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import theano.tensor as T
from theano import function

def k_old(X,Y,gamma):
  return ( 1 / T.pow( 
    2 * pi * T.pow(gamma,2),
  Y.shape[-1]/2) ) * T.exp( 
    -T.pow(X-Y, 2) / (2 * T.pow(gamma, 2) ) 
  ) 
  
def k(X,Y,gamma):
  #return (1/T.pow( gamma * sqrt(2*pi), Y.shape[-1]) ) * T.exp( -T.pow(X-Y,2) / ( 2*T.pow(gamma,2) ) )
  return T.exp(-T.pow(X-Y,2)/(2*gamma**2))
  
col = T.TensorType('float32', [False, False,False])
row = T.TensorType('float32', [False,False])

observations = col('parzen observations')     # [distribution][observation][dimension]
test_points = row('parzen test points')       # [test point][dimension]
gamma = T.fscalar('gamma')                    # value

probabilities = T.sum( 
  T.prod( 
    k(observations.dimshuffle(0,1,'x',2), test_points.dimshuffle('x','x',0,1), gamma),  # => [distribution][observation][test_point][dimension
  3), # => [distribution][observation][test_point]
1) / observations.shape[1]  # => [distribution][test point]
  
parzen_function = function( [observations, test_points, gamma], probabilities )

# Calculate the estimated probability of a set of test_points given a set of observations
def from_one(observations, test_points, gamma):
  if observations.ndim != 2:
    raise ValueError, 'observations must be a 2-dimensional array in the form [sample point][dimension], not shape %s' % str(observations.shape)
  elif test_points.ndim != 2:
    raise ValueError, 'test_points must be a 2-dimensional arran in the form [test point][dimension], not shape %s' % str(test_points.shape)
    
  return parzen_function( np.expand_dims(observations, axis=0), test_points, gamma)
  
  
# Calculate the estimated probability of a set of test_points given multiple sets of observations
def from_many( observation_sets, test_points, gamma):
  if observation_sets.ndim != 3:
    raise ValueError, 'observation sets must be a 3-dimensional array in the form [set][sample point][dimension], not shape %s' % str(observation_sets.shape)
  elif test_points.ndim != 2:
    raise ValueError, 'test_points must be a 2-dimensional arran in the form [test point][dimension], not shape %s' % str(test_points.shape)
    
  return parzen_function( observation_sets, test_points, gamma)

def graph(observations, gamma, xN, yN):
  xrange = [observations[:,0].min(), observations[:,0].max()]
  yrange = [observations[:,1].min(), observations[:,1].max()]
  xstep = ((xrange[1]-xrange[0] ) / xN )
  ystep = ((yrange[1]-yrange[0] ) / yN )
  X = np.dstack(np.mgrid[xrange[0]:xrange[1]:xstep,yrange[0]:yrange[1]:ystep]).reshape([ xN *yN,2])
  x = np.arange(xrange[0],xrange[1],xstep)
  y = np.arange(yrange[0],yrange[1],ystep)

  # [test point][dimension]
  test_points = X.astype('float32').reshape(xN*yN,2)

  # NOTE: the PDF contours seem to be *displaying* properly (not being computed properly though)
  pdf = from_one(observations,test_points,gamma)
  observation_probability = from_one(observations,observations,gamma)

  z = pdf.reshape([xN,yN]).T
  #sizes = observation_probability.reshape(observation_probability.shape[0])
  #sizes = sizes * ( 50/sizes.max() )


  CS = plt.contour(x,y,z,10)
  plt.scatter( observations[:,0], observations[:,1] )#, sizes ) # NOTE: these points seem to be displaying properly
  plt.clabel(CS, inline=1, fontsize=10)
  plt.axis( [xrange[0],xrange[1],yrange[0],yrange[1]] )
  plt.show()