#! /usr/bin/env python

# Based on three papers:
# (1) Support Vector Density Estimation, and
# (2) Density Estimation using Support Vector Machines
# (3) An Improved Training Algorithm for Support Vector Machines
# (1) and (2) by Weston et. al, the first from '99 and the second from '98
# They seem to be reprints of the same paper
# (3) By Osuna et. al.

# Specifically, this is intended to implement the decomposition algorithm 
# described in (3) using the Support Vector Machine described in (1)-1.9

# NOTE: The general approach here is going to be to implement the SVM
# first, then work out the math for decomposing it.

import sys, getopt, math, datetime, os, cmath
from random import gauss

import numpy
import scipy
import scipy.special
import scipy.stats
import cvxopt
#import cvxmod
from cvxopt import *

from numpy import *

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['abstol'] = 1e-15
cvxopt.solvers.options['reltol'] = 1e-15
cvxopt.solvers.options['feastol'] = 1e-15

_Functions = ['run']
	
class svm:
	def __init__(self,data=list(),Lambda=.1, gamma =arange(1,4,1) ):
	# SVM Class
	#
	# @param data		[Nxd] array of observations where N is the number of observations and d is the dimensionality of the abstract space
	# @param Lambda		Regularizer to control Smoothness / Accuracy.  Preliminary experimental results show the range 0-1 controls this parameter.
	# @param gamma		List of gamma values which define the kernel smoothness
	
		try:
			self.N,self.d = data.shape
		except ValueError:
			self.N,self.d = (len(self.X),1)
			self.X = data.reshape([ self.N, self.d ])
		else:
			self.X = data
		
		self.Lambda = Lambda
		self.gamma = gamma
		
		self.Gamma = None		# gamma repeated N times
		self.Y = None				# empirical CDF of X
		self.SV = None			# X value array of SV
		self.NSV = None			# cardinality of SV
		self.alpha = None			# the full weight array for all observations
		self.beta = None			# weight array for SV
		self.K = None				# precomputed kernel matrix
		
		self._compute()
		
	def __str__(self):
		ret = "SVM Instance\n"
		ret += "X: [%s x %sd]\n" % (self.N, self.d)
		ret += "Lambda: %s\n" % self.Lambda
		ret += "gamma: %s\n" % str(self.gamma)
		ret += "SV: %s (%s percent)\n" % ( self.NSV,100. * float(self.NSV) / float(self.N ) )
		ret += "Loss: %s\n" % (self.cdf_res()**2).sum()
		return ret
	
	def _Omega(self,Gamma):
	# Regularizer function
	#
	# @param Gamma			[N*kappa x 1] array of gamma values
	
		return self.Lambda * (Gamma)
		
	def _K(self,X,Y,gamma):
	# Kernel function
	#
	# @param X				[Nxd] array of observations
	# @param Y				[Mxd] array of observations
	# @param gamma			kernel width parameter
	
		N,d1 = X.shape
		M,d2 = Y.shape
		
		if d1 != self.d != d2:
			raise StandardError, 'Matrices do not conform to the dimensionality of existing observations'
		
		diff = X.reshape([N,1,self.d]) - numpy.transpose( Y.reshape([M,1,self.d]), [1,0,2] )
		
		# Sigmoid
		return ( 1.0 / ( 1.0 + numpy.exp( -gamma * diff ) ) ).prod(2).reshape(N,M)
		
		# RBF
		#return ( exp( -((X-Y)**2.0) / gamma ) ).reshape(N,M)

	def _k(self,X,Y,gamma):
	# Cross-kernel function
	#
	# @param X				[Nxd] array of observations
	# @param Y				[Mxd] array of observations
	# @param gamma			kernel width parameter
	
		N,d1 = X.shape
		M,d2 = Y.shape
		
		if d1 != self.d != d2:
			raise StandardError, 'Matrices do not conform to the dimensionality of existing observations'
		
		diff = X.reshape([N,1,self.d])- numpy.transpose( Y.reshape([M,1,self.d]), [1,0,2] )
		
		return ( gamma / ( 2.0 + numpy.exp( gamma * diff ) + numpy.exp( -gamma * diff ) ) ).prod(2).reshape(N,M)
		
	def cdf(self,X):
	# Cumulative distribution function
	#
	# @param X				[Nxd] array of points for which to calculate the CDF
		
		return numpy.dot( self._K( X, self.SV, self.Gamma ), self.beta )
		
	def pdf(self,X):
	# Probability distribution function
	#
	# @param X				[Nxd] array of points for which to calculate the PDF
	
		return numpy.dot( self._k( X, self.SV, self.Gamma ), self.beta )
		
	def cdf_res(self,X=None):
	# CDF residuals
	#
		if X==None:
			X = self.X
		#return ( self.Y.flatten() - self.cdf( X.flatten() ).flatten() )
		return self.Y - atleast_2d(X)
		
	def __iadd__(self, points):
		# overloaded '+=', used for adding a vector list to the module's data
		# 
		# @param points		A LIST of observation vectors (not a single ovservation)
		raise StandardError, 'Not Implemented'
		self.X += points
	
	def _compute(self):
		start = datetime.datetime.now()
		
		kappa = len( self.gamma )
		(N,self.d) = self.X.shape
		# product CDF
		self.Y = ( ( .5 + (self.X.reshape(N,1,self.d) > transpose(self.X.reshape(N,1,self.d),[1,0,2])).prod(2).sum(1,dtype=float) ) / N ).reshape([N,1])
		# sum CDF
		#self.Y = ( ( .5 + (self.X.reshape(N,1,self.d) > transpose(self.X.reshape(N,1,self.d),[1,0,2])).sum(2).sum(1,dtype=float) ) / (N*self.d) ).reshape([N,1])
		self.K = numpy.hstack(  [self._K( self.X, self.X, gamma ) for gamma in self.gamma] )
		self.Gamma = numpy.repeat(self.gamma,N).reshape([N*kappa,1])
		
		P = cvxopt.matrix( numpy.dot(self.K.T,self.K), (N*kappa,N*kappa) )
		q = cvxopt.matrix( ( self._Omega(self.Gamma) - ( numpy.ma.dot( self.K.T, self.Y ) ) ), (N*kappa,1) )
		G = cvxopt.matrix( -identity(N*kappa), (N*kappa,N*kappa) )
		h = cvxopt.matrix( 0.0, (N*kappa,1) )
		A = cvxopt.matrix( 1., (1,N*kappa) )
		b = cvxopt.matrix( 1., (1,1) )
		
		# Solve!
		p = solvers.qp( P=P, q=q, G=G, h=h, A=A, b=b )
		
		beta = ma.masked_less( p['x'], 1e-8 )
		mask = ma.getmask(beta)
		self.NSV = beta.count()
		self.alpha = beta
		self.beta = beta.compressed().reshape([self.NSV,1])
		self.SV = numpy.ma.array( numpy.tile(self.X.T,kappa).T, mask=numpy.repeat(mask,self.d)).compressed().reshape([self.NSV,self.d])
		self.Gamma = numpy.ma.array( self.Gamma, mask=mask ).compressed().reshape([self.NSV,1])

		duration = datetime.datetime.now() - start
		print "optimized in %ss" % ( duration.seconds + float(duration.microseconds)/1000000)
		
def run():
	fig = plt.figure()
	
	samples = numpy.random.multivariate_normal( mean=array([5,5,5]), cov=array( numpy.identity(3) ), size=array([100,]) )
	
	mod = svm( samples, Lambda=1e-8, gamma=[.5,1] )
	
	print mod
	
	#X = dstack(mgrid[0:10,0:10]).reshape([100,2])
	X = numpy.vstack( [ arange(0,10,.1), 5*numpy.ones([100,]), 5*numpy.ones([100,]) ] ).T
	
	#plt.contourf(arange(0,10),arange(0,10),mod.pdf(X).reshape([10,10]), antialiased=True, colors='k' )
	#plt.plot( hsplit(samples,2)[0],hsplit(samples,2)[1], '+' )

	plt.plot( arange(0,10,.1), mod.pdf(X) )
	plt.plot( arange(0,10,.1),mod.cdf(X), '--' )

	#a = fig.add_subplot(2,2,1)
	#a.plot( [ i % mod.N for i in range( mod.N * len(mod.gamma) ) ], mod.alpha, 'o' )
	#a.set_title("weights (x=ell)")
	
	#b = fig.add_subplot(2,2,3)
	
	#c = fig.add_subplot(2,2,2)
	
	#d = fig.add_subplot(2,2,4)
	
	plt.show()
	
	
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
