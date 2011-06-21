#! /usr/bin/env python

import sys, getopt, math, datetime, os
sys.path.append('../')
from random import gauss

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#from mpl_toolkits.mplot3d import Axes3D
import enthought.mayavi.mlab as mlab
import scikits.statsmodels.api as sm
import scipy as sp
#from scikits.learn.svm import *

_Functions = ['run', 'test_system4', 'test_system3', 'test_parzen', 'test_divergence']
	
import theano.tensor as T
from theano import function

from a_machine.gpu_funcs import kernel_matrix
from a_machine.svms import SVR

def run():
  print "Initializing"
  
  from a_machine.system4 import model

  from santa_fe import getData
  data = getData('B1.dat')
  test = getData('B2.dat')
  median = np.median(data[:,:2], axis=0)
  std = np.std(data[:,:2], axis=0)

  train_size = 1000
  sequence_length = 1
  gamma_quantile = 100
  test_size = 200
    
  train = (( data[:train_size,:2] - median ) / std).astype('float32')
  labels = (( data[sequence_length:train_size+sequence_length,0] - median[0] ) / std[0]).astype('float32')
  
  
  #svm = SVR( nu=.1, C=50)
  m = model( dimension=0, sequence_length=sequence_length )
  #svm.train( kernel_matrix(np.expand_dims(train,1), np.expand_dims(train,1), .5), labels)
  m.train(train[:,:2], train_size)

  xN = 100
  yN = 100
  xrange = [train[:,0].min(), train[:,0].max()]
  yrange = [train[:,1].min(), train[:,1].max()]
  xstep = ((xrange[1]-xrange[0] ) / xN )
  ystep = ((yrange[1]-yrange[0] ) / yN )
  X = np.dstack(np.mgrid[xrange[0]:xrange[1]:xstep,yrange[0]:yrange[1]:ystep]).reshape([ xN *yN,2]).astype('float32')
  x = np.arange(xrange[0],xrange[1],xstep)
  y = np.arange(yrange[0],yrange[1],ystep)

  #Z = svm.predict( kernel_matrix( np.expand_dims(X,1), np.expand_dims(train,1), .5) )
  Z = m.predict( np.expand_dims(X,1) )
  
  #ax = plt.subplot(111, projection='3d')
  #ax.plot( x[:,0], x[:,1], y, 'k,' )
  #plt.plot( train[:,0], labels, 'k,', alpha=.2 )
  #plt.plot( train[svm.support_,0], labels[svm.support_], 'o', alpha=.15 )
  #plt.plot(x[:,0],y, 'r', lw=2)
  #plt.show()
  
  mlab.points3d(train[:,0], train[:,1], labels, scale_factor=.05, opacity=.2)
  mlab.points3d(train[m.svm.SV_indices,0], train[m.svm.SV_indices,1], labels[m.svm.SV_indices], scale_factor=.05)
  mlab.surf( x,y,Z.reshape(xN,yN) )

  
  print "%s SV of %s" % (len(m.svm.SV_indices), train.shape[0])
  
  mlab.show()
  
def test_system4():
  print "Initializing"
  
  train_size = 1000
  sequence_length = 1
  gamma_quantile = 100
  test_size = 200

  import a_machine.system4 as system
  
  print "Importing & Normalizing Data"
  
  from santa_fe import getData
  data = getData('B1.dat')
  test = getData('B2.dat')
  median = np.median(data, axis=0)
  std = np.std(data, axis=0)

  # normalizing to median 0, std deviation 1
  normed_data = ( data - median ) / std
  
  print "Initializing Models"
  
  model = system.model(dimension=0, gamma_samples=1000, gamma_quantile=gamma_quantile, sequence_length=sequence_length) 
  model.train( normed_data, train_size )   
  
  print "Generating Predictions"
  
  # [test_point][dimension]
  test = test[:test_size+sequence_length,:]
  #test = data[:test_size+sequence_length,:]
  normed_test = (test - median) / std
  predictions = model.predict(normed_test)
  
  # denormalize
  #predictions = ( std[0] * predictions ) + median[0]

  errors = np.abs( normed_test[sequence_length : predictions.shape[0]+sequence_length, 0] - predictions )
  
  print "Results!  Loss/point: %s (in normed space)" % ( errors.sum(0) / test_size )
  
  x = np.arange( predictions.shape[0] )
  
  x_pred = np.dstack( [
    np.arange(model.sequences[:,0,0].min(), model.sequences[:,0,0].max(), ( model.sequences[:,0,0].max() - model.sequences[:,0,0].min() ) / 100 ),
    np.arange(model.sequences[:,0,1].min(), model.sequences[:,0,1].max(), ( model.sequences[:,0,1].max() - model.sequences[:,0,1].min() ) / 100 ),
    np.arange(model.sequences[:,0,2].min(), model.sequences[:,0,2].max(), ( model.sequences[:,0,2].max() - model.sequences[:,0,2].min() ) / 100 ),
  ]).astype('float32').reshape(100,1,3)

  y_pred = model.svm.predict( kernel_matrix(x_pred, model.sequences, model.gammas[-1] ) )
  print x_pred[0]
  print x_pred[1]
  
  pr = plt.subplot(2,2,1)
  pr.plot(x,normed_test[sequence_length : predictions.shape[0]+sequence_length, 0], 'k', alpha=.4)
  #for i in range(predictions.shape[1]):
  #  for j in range(predictions.shape[2]):
  #    plt.plot(x,predictions[:,i,j])
  pr.plot(x,predictions)
  
  
  reg0 = plt.subplot(2,2,2)
  reg0.plot(model.sequences[:,0,0], model.labels, 'k,', alpha=.5) 
  reg0.plot(model.sequences[model.svm.SV_indices,0,0], model.labels[model.svm.SV_indices], 'o', alpha=.15 )
  reg0.plot(x_pred[:,0,0], y_pred, 'r', lw=2)

  reg1 = plt.subplot(2,2,3)
  reg1.plot(model.sequences[:,0,1], model.labels, 'k,', alpha=.5) 
  reg1.plot(model.sequences[model.svm.SV_indices,0,1], model.labels[model.svm.SV_indices], 'o', alpha=.15 )
  reg1.plot(x_pred[:,0,1], y_pred, 'r',lw=2)

  reg2 = plt.subplot(2,2,4)
  reg2.plot(model.sequences[:,0,2], model.labels, 'k,', alpha=.5) 
  reg2.plot(model.sequences[model.svm.SV_indices,0,2], model.labels[model.svm.SV_indices], 'o', alpha=.15 )
  reg2.plot(x_pred[:,0,2], y_pred, 'r',lw=2)
    
  plt.show()

  return
  
def test_system3():
  print "Initializing"
  
  train_size = 500
  sequence_length = 2
  gamma_quantile = 50
  test_size = 500

  import a_machine.system3 as system
  
  print "Importing & Normalizing Data"
  
  from santa_fe import getData
  data = getData('B1.dat')
  test = getData('B2.dat')
  median = np.median(data, axis=0)
  std = np.std(data, axis=0)

  # normalizing to median 0, std deviation 1
  data = ( data - median ) / std
  
  
  print "Initializing Models"
  
  model = system.model(gamma_samples=1000, gamma_quantile=gamma_quantile, sequence_length=sequence_length) 
  model.train( data, train_size )   
  
  print "Generating Predictions"
  
  # [test_point][dimension]
  #normed_test = (test[:test_size,:] - median) / std
  normed_test = data[:test_size]
  predictions, risks = model.predict(normed_test)
  hybrid = ( predictions * risks ).sum(1) / risks.sum(1)
  
  # denormalize
  predictions = ( std.reshape(1,1,3) * predictions ) + median.reshape(1,1,3)
  hybrid = ( std.reshape(1,3) * hybrid ) + median.reshape(1,3)
  
  print "Results!"
  
  errors = np.abs( np.expand_dims( test[sequence_length : test_size], 1) - predictions )
  hybrid_error = np.abs( test[sequence_length : test_size] - hybrid )
  print hybrid.shape
  print hybrid_error.shape
  
  print ( hybrid_error.sum(0) / test_size )
  print ( errors.sum(0) / test_size )
  print std.astype('int')
  
  x = np.arange(test_size-sequence_length)
  
  for i in range(data.shape[1]):
    fig = plt.subplot(data.shape[1],1,i+1)
    
    fig.plot(x, test[sequence_length : test_size,i], 'k--')
    for j in range(predictions.shape[1]):
      fig.plot(x, predictions[:,j,i] )

    fig.plot(x, hybrid[:,i], 'r', lw=2)
  plt.show()

  
def test_divergence():
  print "Starting"
  
  import cs_divergence, parzen_probability
  
  print "compiled for GPU"
  
  xrange = [0,1]
  xstep = .01
  xN = int((xrange[1]-xrange[0])/xstep)
  x=np.arange(xrange[0],xrange[1],xstep).astype('float32')
  gamma = .1
  distN = 20
  baseN = 20
  
  # 5 distributions containing 5 1-d points
  distributions = np.array( [
    np.random.normal(.2, .05, distN), 
    np.random.normal(.4, .05, distN), 
    np.random.normal(.6, .05, distN),  
    np.random.normal(.8, .05, distN)
  ] ).reshape(4,distN,1).astype('float32')
  # distribution with 5 1-d points
  base = np.random.normal(.8, .05, baseN).reshape(baseN,1).astype('float32')
  
  divergences = cs_divergence.from_many(distributions, base, gamma=gamma)
  
  for i in range(4):
    ax = plt.subplot(2,2,i+1, title="Divergence: %s" % divergences[i])
    
    ax.plot(x, parzen_probability.from_many( distributions[i].reshape(1,distN,1), x.reshape(xN,1), gamma=gamma ).reshape(xN), 'b' )
    ax.plot(x, parzen_probability.from_many( base.reshape(1,baseN,1), x.reshape(xN,1), gamma=gamma ).reshape(xN), 'g--' )
    ax.axis([0,1,0,None])
  
  plt.show()
  
  
def test_parzen():
	print "Starting"
	
	import parzen_probability
	
	print "compiled for GPU"
	
	parzen_probability.graph(np.random.rand(10,2).astype('float32'), .1, 100, 100)
  
def help():
	print __doc__
	return 0
	
def process(arg='run'):
	if arg in _Functions:
		globals()[arg]()
	
class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(argv=None):
	if argv is None:
		argv = sys.argv
	try:
		try:
			opts, args = getopt.getopt(sys.argv[1:], "hl:d:", ["help","list=","database="])
		except getopt.error, msg:
			raise Usage(msg)
		
		# process options
		for o, a in opts:
			if o in ("-h", "--help"):
				for f in _Functions:
					if f in args:
						apply(f,(opts,args))
						return 0
				help()
		
		# process arguments
		for arg in args:
			process(arg) # process() is defined elsewhere
			
	except Usage, err:
		print >>sys.stderr, err.msg
		print >>sys.stderr, "for help use --help"
		return 2

if __name__ == "__main__":
	sys.exit(main())
