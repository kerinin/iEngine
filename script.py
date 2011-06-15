#! /usr/bin/env python

import sys, getopt, math, datetime, os
sys.path.append('../')
from random import gauss

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scikits.statsmodels.api as sm
import scipy as sp

_Functions = ['run', 'test_parzen', 'test_divergence']
	
import theano.tensor as T
from theano import function

def run():
  print "Initializing"
  
  gamma_increment = 50
  gamma_samples = 1000
  sequence_length = 1
  train_size = 10000
  test_size = 500

  import a_machine.system3 as system
  
  print "Importing & Normalizing Data"
  
  from santa_fe import getData
  data = getData('B1.dat')[:train_size,:]
  test = getData('B2.dat')
  median = np.median(data, axis=0)
  std = np.std(data, axis=0)

  # normalizing to median 0, std deviation 1
  data = ( data - median ) / std


  print "Dermining gamma values"
  
  g_samples = data.copy()
  np.random.shuffle(g_samples)
  g_samples = g_samples[:gamma_samples]
  g_diff = np.abs( g_samples.reshape(g_samples.shape[0],1,g_samples.shape[1]) - g_samples.reshape(1,g_samples.shape[0],g_samples.shape[1]) )
  g_diff = g_diff.reshape(g_samples.shape[1]*g_samples.shape[0]**2)
  g_percentiles = np.arange(gamma_increment / 2,100,gamma_increment).astype('float')
  gammas = []
  for i in g_percentiles:
    gammas.append( sp.stats.stats.scoreatpercentile(g_diff, i) ) 
    
  print "--> %s gamma values: %s" % (len(gammas), str(gammas))
  #ecdf = sm.tools.tools.ECDF(g_diff)
  #x = np.linspace(min(g_diff), max(g_diff))
  #y = ecdf(x)
  #plt.step(x, y)
  #plt.plot(gammas, np.array(percentiles)/100, 'o')
  #plt.show()
  
  
  print "Initializing Models"
  
  models = []
  for gamma in gammas:
    model = system.model(gamma, sequence_length)
    model.train(data)
    models.append(model)
    
  
  print "Generating Predictions"
  
  # [model][test_point][dimension]
  normed_test = (test[:test_size,:] - median) / std
  predictions = []
  for model in models:
    predictions.append(model.predict(normed_test))
  predictions = np.array(predictions)
  
  
  # denormalize
  predictions = ( std.reshape(1,1,3) * predictions ) + median.reshape(1,1,3)
  #print boc
  

  print "Results!"
  
  
  error = np.abs( np.expand_dims( test[sequence_length : test_size], 0) - predictions )
  print error.sum(1) / test_size
  print std
  
  plt.plot(np.arange(test_size-sequence_length), test[sequence_length : test_size,0], 'k')
  plt.plot(np.arange(test_size-sequence_length), predictions[0,:,0], 'g--')
  plt.plot(np.arange(test_size-sequence_length), predictions[1,:,0], 'r--')
  plt.show()

  
  #for i in range(data.shape[1]):
    #fig = plt.subplot(data.shape[1], 1, i+1)
    #fig.hist(error[:,i])
    
  #plt.show()
  
  
  

  return
  #sequences = series.reshape(series.shape[0],1) + np.zeros((1,3))
  #sequences[:-1,1] = sequences[1:,1]
  #sequences[:-2,2] = sequences[2:,2]
  #sequences[:-2,2] = data[2:,2]
  #sequences = sequences[:-2,:]
  
  #print sequences[:,0]
  #print sequences[:,1]
  #mask1 = np.ma.masked_less( sequences[:,2], median - std).mask
  #mask2 = np.ma.masked_inside( sequences[:,2], median - std, median).mask
  #mask3 = np.ma.masked_inside( sequences[:,2], median, median + std).mask
  #mask4 = np.ma.masked_greater( sequences[:,2], median + std).mask
  
  #plt.plot(np.ma.array(sequences[:,0], mask=mask1), np.ma.array(sequences[:,1], mask=mask1), 'ro', alpha=.05 )
  #plt.plot(np.ma.array(sequences[:,0], mask=mask2), np.ma.array(sequences[:,1], mask=mask2), 'go', alpha=.05 )
  #plt.plot(np.ma.array(sequences[:,0], mask=mask3), np.ma.array(sequences[:,1], mask=mask3), 'bo', alpha=.05 )
  #plt.plot(np.ma.array(sequences[:,0], mask=mask4), np.ma.array(sequences[:,1], mask=mask4), 'yo', alpha=.05 )
  #plt.show()
  #return
  
  #fig = plt.figure()
  #a = fig.add_subplot(3,1,1)
  #a.plot( np.arange(0,400), data[:400,0])
  #b = fig.add_subplot(3,1,2)
  #b.plot( np.arange(0,400), data[:400,1])
  #c = fig.add_subplot(3,1,3)
  #c.plot( np.arange(0,400), data[:400,2])
  #plt.show()
  #return
  
  print "Data Imported"
  
  # use the scaled standard deviation as a starting point for the dimension-wise gamma
  gamma = np.std(data, axis=0) / 10
  m = system.model(sequence_length, gamma)
  
  print "Model Built"
  
  accuracies = []
  for i in range(iterations):
    m.process( data[i*training_increment : (i+1)*training_increment] )
    
    accuracies.append( map( lambda x: test[x+sequence_length+1] - m.predict_from(test[x:x+sequence_length]), np.arange(500,600,10) ) )
  accuracies = np.absolute( np.array(accuracies) ).sum(1)
  
  print "Predictions done"
  
  X = np.arange(0,iterations * training_increment, training_increment)
  plt.plot(X, accuracies[:,2] )
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
