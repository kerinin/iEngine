#! /usr/bin/env python



import numpy as np
import numpy.ma as ma
import scipy
import scipy.special
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm

from SVM_D3 import svm

_Functions = ['single',]

def single():
	samples1 = vstack( [ 
		numpy.random.multivariate_normal( mean=array([0,0]), cov=array( identity(2) ), size=array([50,]) ),
		numpy.random.multivariate_normal( mean=array([0,6]), cov=array( identity(2) ), size=array([50,]) ) 
	] )
	samples2 = vstack( [ 
		numpy.random.multivariate_normal( mean=array([0,0]), cov=array( identity(2) ), size=array([50,]) ),
		numpy.random.multivariate_normal( mean=array([6,0]), cov=array( identity(2) ), size=array([50,]) ) 
	] )
	
	phi_1 = svm( samples1, Lambda=.005, gamma=[.125,.25,.5,1,2,4,8,16] )
	phi_2 = svm( samples2, Lambda=.005, gamma=[.125,.25,.5,1,2,4,8,16] )
	
	fig = plt.figure()
	
	a = fig.add_subplot(2,2,1)
	phi_1.contourPlot( fig=a, xrange=(-2,8), yrange=(-2,8), xstep=.1, ystep=.1, title="phi_1 distribution" )

	b = fig.add_subplot(2,2,1)
	phi_2.contourPlot( fig=b, xrange=(-2,8), yrange=(-2,8), xstep=.1, ystep=.1, title="phi_2 distribution" )
	
	
	
	plt.show()
	
	
def help():
	print __doc__
	return 0
	
def process(arg='single'):
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
