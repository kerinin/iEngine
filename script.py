#! /usr/bin/env python

import sys, getopt, math, datetime, os
sys.path.append('../')
from random import gauss

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_Functions = ['run', 'test_parzen', 'test_divergence']
	
import theano.tensor as T
from theano import function

def run():
  print "Starting"
  
  sequence_length = 20
  training_increment = 200
  iterations = 20
  
  import a_machine.system2 as system
  
  print "Compiled for GPU"
  
  from santa_fe import getData
  data = getData('B1.dat')
  test = getData('B2.dat')
  
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
