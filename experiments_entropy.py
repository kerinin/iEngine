#! /usr/bin/env python

import sys, getopt, math, datetime, os, cmath
from random import gauss

import numpy as np
import numpy.ma as ma
import scipy
import scipy.special
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm

from SVM_D2 import svm
from engine import engine

_Functions = ['single',]

def single():
	
	fig = plt.figure()
	xrange=(-3.,9.)
	yrange=(-3.,9.)
	xstep=.1
	ystep=.1
	
	samples1 = np.vstack( [ 
		np.random.multivariate_normal( mean=np.array([0,0]), cov=np.array( np.identity(2) ), size=np.array([50,]) ),
		np.random.multivariate_normal( mean=np.array([0,6]), cov=np.array( np.identity(2) ), size=np.array([50,]) ) 
	] )
	phi_1 = svm( samples1, Lambda=.005, gamma=[.125,.25,.5,1,2,4,8,16] )
	a = fig.add_subplot(2,2,1)
	phi_1.contourPlot( fig=a, xrange=xrange, yrange=yrange, xstep=xstep, ystep=ystep, title="phi_1 distribution" )
	
	samples2 = np.vstack( [ 
		np.random.multivariate_normal( mean=np.array([0,0]), cov=np.array( np.identity(2) ), size=np.array([50,]) ),
		np.random.multivariate_normal( mean=np.array([6,0]), cov=np.array( np.identity(2) ), size=np.array([50,]) ) 
	] )
	phi_2 = svm( samples2, Lambda=.005, gamma=[.125,.25,.5,1,2,4,8,16] )
	b = fig.add_subplot(2,2,2)
	phi_2.contourPlot( fig=b, xrange=xrange, yrange=yrange, xstep=xstep, ystep=ystep, title="phi_2 distribution" )
	
	e = engine( (phi_1,phi_2) )
	
	test1 = np.vstack( [ 
		np.random.multivariate_normal( mean=np.array([0,0]), cov=np.array( np.identity(2) ), size=np.array([10,]) )
	] )
	c = fig.add_subplot(2,2,3)
	e.contourPlot( S=test1, fig=c, xrange=xrange, yrange=yrange, xstep=xstep, ystep=ystep, title="derived distribution 1" )
	
	test2 = np.vstack( [ 
		np.random.multivariate_normal( mean=np.array([0,6]), cov=np.array( np.identity(2) ), size=np.array([10,]) )
	] )
	c = fig.add_subplot(2,2,4)
	e.contourPlot( S=test2, fig=c, xrange=xrange, yrange=yrange, xstep=xstep, ystep=ystep, title="derived distribution 2" )
	
	
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
