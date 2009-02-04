#! /usr/bin/env python

import sys, getopt, math, datetime, os, cmath
from random import gauss

import numpy
import scipy
import scipy.special

from numpy import *
#from pylab import *

_Functions = ['run']
	
class svm:
	def __init__(self,data=list(),gamma=None):
		self.data = data
		self.SV = list()
		self.gamma = gamma

		if gamma:
			self.param.gamma = gamma
		
		self._compute()
	
	def __iadd__(self, points):
		# overloaded '+=', used for adding a vector list to the module's data
		# 
		# @param points		A LIST of observation vectors (not a single ovservation)
		self.data += points
	
	def _compute(self):
		C = 1
		epsilon = .5
		
		# Set the training pairs
		# (x_1,F_N(x_1)),...,(x_N,F_N(x_N))
		# F_N = 1/N \sum_{k=1}^N I(x-x_k)
		# I = Indicator function (0 if negative, 1 otherwise)
		t = ( ( self.data.T > self.data ).sum(1,dtype=float) / len(self.data) ).reshape( len(self.data), 1)	# Since F is an estimate of the target value t, i'm simply renaming it
		
		# Set learning rate \rho and randomly set w_i
		rho = 1e-4
		W = numpy.random.rand( len(self.data), 1)
		
		# Calculate covariance matrix K and let \sigma_i^2 = K_{ii}
		# K(x_i,x_j) = 1/(\sqrt(2 \pi det( \Lambda ) ) ) exp( -.5(x_i - x_j) \Lambda^-1 (x_i - x_j)^T
		# \Lambda = variable (gamma)
		# \sigma = K_{ii}
		Lambda = .5
		K = 1 / ( sqrt( 2 * Lambda ) )* exp( -.5 * ( self.data - self.data.T ) * ( self.data-self.data.T).T / Lambda ) 
		sigma2 = K.diagonal().reshape( len(self.data), 1)
		
		def inner(K,W,C,epsilon,sigma2):
			# Inner Loop
			# yX = \langle y(x) \rangle = \sum_{i=1}^N w_i K(x, x_i )
			yX = dot(K[0].reshape( 1,len(self.data) ),W)[0][0]
			
			# yXi = \langle y(s) \rangle_i = yX - \sigma_i^2 w_i
			yXi = yX - sigma2 * W
			
			# Fi = C/2 exp( C/2 ( 2yXi - 2t_i + 2\epsilon + C \sigma_i^2 ) )
			#	* ( 1 - erf( ( yXi - t_i + \epsilon + C \sigma_i^2 ) / sqrt( 2 \sigma_i^2 ) )
			#	- C/2 exp( C/2 ( -2yXi + 2 t_i + 2 \epsilon + C \sigma_i^2 ) )
			#	* ( 1 - erf( -C/2 ( 2yXi - 2t_i + 2\epsilon + C \sigma_i^2 ) )

			Fi = (
				C/2. * exp( (C/2.) * ( ( 2. * yXi ) - ( 2. * t ) + ( 2. * epsilon ) + ( C * sigma2 ) ) ) 
				* ( 1. - scipy.special.erf( ( yXi - t + epsilon + ( C * sigma2 ) ) / sqrt( 2. * sigma2 ) ) )
				- C/2. * exp( (C/2.) * ( ( -2. * yXi ) + ( 2. * t ) + ( 2. * epsilon ) + sigma2 ) )
				* ( 1. - scipy.special.erf( ( -C / 2. ) * ( ( 2. * yXi ) - ( 2. * t ) + ( 2. * epsilon ) + ( C * sigma2 ) ) ) )
				)
				
			# Gi = 1/2 erf( ( t_i - yXi + \epsilon ) / sqrt( s \sigma_i^2 ) )
			#	- 1/2 erf( ( t_i - yXi - \epsilon ) / sqrt( s \sigma_i^2 ) )
			#	+ 1/2 exp( C/2 ( 2 yXi - 2 t_i + 2 \epsilon + C \sigma_i^2 ) )
			#	* ( 1 - erf( ( yXi - t_i + \epsilon + \C \sigma_i^2 ) / sqrt( 2 \sigma_i^2 ) )
			#	+ 1/2 exp( C/2 ( -2 yXi - 2 t_i + 2 \epsilon + C \sigma_i^2 ) )
			#	* ( 1 - erf( ( yXi - t_i + \epsilon + \C \sigma_i^2 ) / sqrt( 2 \sigma_i^2 ) )
			Gi = (
				.5 * scipy.special.erf( ( t - yXi + epsilon ) / sqrt( 2. * sigma2 ) )
				- .5 * scipy.special.erf( ( t - yXi - epsilon ) / sqrt( 2. * sigma2 ) )
				+ .5 * exp( (C/2.) * ( ( 2. * yXi ) - ( 2. * t ) + ( 2. * epsilon ) + ( C * sigma2 ) ) )
				* ( 1. - scipy.special.erf( ( yXi - t + epsilon + ( C * sigma2 ) ) / sqrt( 2. * sigma2 ) ) )
				+ .5 * exp( (C/2.) * ( ( -2. * yXi ) - ( 2. * t ) + ( 2. * epsilon ) + ( C * sigma2 ) ) )
				* ( 1. - scipy.special.erf( ( yXi - t + epsilon + ( C * sigma2 ) ) / sqrt( 2. * sigma2 ) ) )
				)

			# erf(x) = 2/sqrt(\pi) \sum_0^x e^{-t^2} dt (see scipy.special.erf)
			# w_i = w_i + \rho ( ( F_i / G_i ) - w_i )
			return rho * ( Fi / Gi - W )
				
		start = datetime.datetime.now()
		while True:
			for i in range(1000):
				W += inner(K=K,W=W,C=C,epsilon=epsilon,sigma2=sigma2)
			W_delta = inner(K=K,W=W,C=C,epsilon=epsilon,sigma2=sigma2)
			if absolute(W_delta).sum() < 1e-6:
				break
			else:
				print "Cumulative adjustment of Coefficients: %s" % absolute(W_delta).sum()
				W += W_delta
				
		print "*** Optimization completed in %ss" % (datetime.datetime.now() - start).seconds
		print W
		
def run():
	mod = svm( array([[gauss(1.,.5)] for i in range(40) ]).reshape([40,1]) )
	

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
