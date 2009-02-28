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

# Decomposition options:
# (A) Fast Training of Support Vector Machines using Sequential Minimal Optimization
# (B) Working Set Selection Using Second Order Information for Training Support Vector Machines
# (C) An Improved Training Algorithm for Support Vector Machines
# (D) Parallel Support Vector Machines:  The Cascade SVM
# (E) A parallel training algorithm for large scale support vector machines
# (F) CUSVM: A CUDA IMPLEMENTATION OF SUPPORT VECTOR CLASSIFICATION AND REGRESSION
# (G) Fast Support Vector Machine Training and Classification on Graphics Processors
# (H) A Study on SMO-type Decomposition Methods for Support Vector Machines


# A Presents SMO as an algorithm (apparently different than Osuna)
# B is implemented in LIBSVM, and essentially presents a new method for determining which coef to optimiza
# C is the original decomposition method, (better explained in A)
# D seems to be parallelized SMO
# E seems to be regular QP with parallelized matrix operations at some steps
# F & G both present CUDA implementations, which in both cases seem to rely on 2nd-order SMO (B)
# H is a survey of decomposition (and supposedly extends the theory to regression)

# My current inclination is to use Osuna's decomposition w/ cunking determined
# by the working size of a CUDA processor.  This will likely require re-implementation
# of the optimization algorithm, so for now simply being able to do the decomposition
# should be sufficient, as it allows SMO in the short term, and CUDA later on if required

# OK, we're using SMO.
# The architecture will need to be revised to separate the selection of workin sets
# the resolving of sub-problems, and the solving (and data maintenance) of the main
# problem.  Ideally, the main problem is defined by the SVM structure, allowing
# simple on-line processing. 



import sys, getopt, math, datetime, os, cmath
from random import gauss

import numpy as np
import numpy.ma as ma
import scipy
import scipy.special
import scipy.stats
import cvxopt as cvx
#import cvxmod
from cvxopt import solvers

#from np import *

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm



cvx.solvers.options['show_progress'] = False
cvx.solvers.options['abstol'] = 1e-15
cvx.solvers.options['reltol'] = 1e-15
cvx.solvers.options['feastol'] = 1e-15

_Functions = ['run']
	
class svm:
	def __init__(self,data=np.empty([1,1]),Lambda=.1, gamma =np.arange(1,4,1), lazy=True ):
	# SVM Class
	#
	# @param data		[Nxd] array of observations where N is the number of observations and d is the dimensionality of the abstract space
	# @param Lambda		Regularizer to control Smoothness / Accuracy.  Preliminary experimental results show the range 0-1 controls this parameter.
	# @param gamma		List of gamma values which define the kernel smoothness
	# @param lazy		If true, computation takes place when PDF or CDF is requested, otherwise takes place when data added
	
		try:
			self.N,self.d = data.shape
		except ValueError:
			self.N,self.d = (len(self.X),1)
			self.X = data.reshape([ self.N, self.d ])
		else:
			self.X = data
		
		self.Lambda = Lambda
		self.gamma = gamma
		self.kappa = len(gamma)
		
		self.Gamma = None		# gamma repeated N times
		self.Y = np.empty([1,1])	# empirical CDF of X
		self.SV = None			# X value array of SV
		self.NSV = 0				# cardinality of SV
		self.alpha = None			# the full weight array for all observations
		self.beta = None			# weight array for SV
		self.P = dict()				# kernel cache
		self.q = None				# q cache
		
		self.lazy = lazy
		
		if not self.lazy:
			self._compute()
		
	def __iadd__(self,X):
		self.X = np.vstack( self.X, X )
		if not self.lazy:
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
	
		(N,d1) = X.shape
		(M,d2) = Y.shape
		
		if d1 != self.d != d2:
			raise StandardError, 'Matrices do not conform to the dimensionality of existing observations'
		
		diff = X.reshape([N,1,d1]) - np.transpose( Y.reshape([M,1,d2]), [1,0,2] )
		
		# Sigmoid
		return ( 1.0 / ( 1.0 + np.exp( -gamma * diff ) ) ).prod(2).reshape(N,M)
		
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
		
		diff = X.reshape([N,1,self.d])- np.transpose( Y.reshape([M,1,self.d]), [1,0,2] )
		
		return ( gamma / ( 2.0 + np.exp( gamma * diff ) + np.exp( -gamma * diff ) ) ).prod(2).reshape(N,M)
		
	def cdf(self,X):
	# Cumulative distribution function
	#
	# @param X				[Nxd] array of points for which to calculate the CDF
	
		self._compute()
		
		return np.dot( self._K( X, self.SV, self.Gamma ), self.beta )
		
	def pdf(self,X):
	# Probability distribution function
	#
	# @param X				[Nxd] array of points for which to calculate the PDF
	
		self._compute()
			
		return np.dot( self._k( X, self.SV, self.Gamma ), self.beta )
		
	def cdf_res(self,X=None):
	# CDF residuals
	#
		self._compute()
			
		if X==None:
			X = self.X
		return self.Y - self.cdf(X)
		
	def cdf_loss(self,X=None):
		
		return sqrt( (self.cdf_res(X)**2).sum(1) )
	
	def _emp_dist(self, X):
	# Empirical Distribution
	#
	# Calculates the empirical distribution of X based on all existing observations
		(N,self.d) = X.shape
		(M,self.d) = self.X.shape
		return ( ( .5 + (X.reshape(N,1,self.d) > transpose(self.X.reshape(M,1,self.d),[1,0,2])).prod(2).sum(1,dtype=float) ) / M ).reshape([N,1])
		
	def _P( self, A, B ):
	# this is used to compute one of two options: 
	# a 2x2 array defined by the working set
	# or a 2xN-2 array defined by the working set and all observations
	#
	# @param A 			[2xd] Array of observations
	# @param B			[Mxd] Array of observations
	
		#if self.P[A[0]][A[1]]:
		#	return self.Q[A[0]][A[1]]
			
		if not A.shape == (2,self.d) or not B.shape[1] == self.d:
			raise StandardError, 'Arguments have incorrect shape - should be 2xd, Nxd'
			
		K = self._K( A, B, self.Gamma )
		
		ret = ma.dot(K.T,K)
		#self.P[A[0]][A[1]] = ret
		
		return ret
	
	def _q( self, x ):
	# this is used to produce a [1xN] array
	
		#if self.q[x]:
		#	return self.q[x]
		
		if not x.shape[1] == self.d:
			raise StandardError, 'Arguments have incorrect shape - should be Nxd'
			
		K = self._K( self.X, x, self.Gamma )
		
		ret = ( self._Omega(self.Gamma) - ( ma.dot( K.T, self.Y ) ) )
		#self.q[x] = ret
		
		return ret
		
	def _grad(self, maskedX ):
	# Gradient of objective function of alpha at unmasked values of X
	#
	# P dot alpha - q 		(calculated only at alpha - so full P not needed)
	#
	# @param maskedX	# an array of X masked at unneeded values
		
		P = self._P( np.vstack( [maskedX,]*self.kappa) ,np.vstack( [self.X,]*self.kappa ) )
		q =  self._q( np.vstack( [maskedX,]*self.kappa ) )
		
		return ma.dot( P, self.alpha ) - q
		
	def _select_working_set(self):
		kappa = len(self.gamma)
		
		I = ma.masked_less( self.alpha, 1e-8 )
		grad = self._grad( ma.array(self.X, mask=ma.getmask(I) ) )
		
		i = (-grad).argmax()
		
		P = self._P( self.X[ i % self.N ].reshape([1,self.d]), np.vstack( [self.X,]*kappa ) )		# P_IJ
		a = ma.masked_less( P[0,i] + np.diagonal(P) - 2*P, 0 )
		
		#NOTE the actual equation is -grad < -grad_i - i'm not sure if inversing the signs and equality operator is a problem
		grad_i =  self._grad( self.X[i % self.N] )
		b = ma.masked_greater(grad, grad_i) - grad_i
		
		j = ( (b**2) / -a ).argmin()
		
		return (i,j)
		
	def _sub_problem(self,i,j):
		X = array([ self.X[i], self.X[j]]).reshape( [2,self.d] )
		
		P = cvxopt.matrix( self._P( X, X ) )
		q = cvxopt.matrix( ( self._q(X) - np.dot(self_P( X, self.X ), self.alpha) ).T )
		G = cvxopt.matrix( -identity(2), (22) )
		h = cvxopt.matrix( 0.0, (N*kappa,1) )
		A = cvxopt.matrix( 1., (1,N*kappa) )
		b = cvxopt.matrix( 1. + self.alpha[i] + self.alpha[j] - self.alpha.sum(), (1,1) )
		
		# Solve!
		p = solvers.qp( P=P, q=q, G=G, h=h, A=A, b=b )
		
		return ( p['x'][0], p['x'][1] )
		
	def _test_stop(self):
		I = ma.masked_less( self.alpha, 1e-8 )
		grad = self._grad(I)
		
		return -grad.max() + grad.min() <= 1e-8
		
	def _compute(self):
		if self.Y.shape != self.X.shape:
			start = datetime.datetime.now()
			
			# Initialize variables
			kappa = len( self.gamma )
			(N,self.d) = self.X.shape
			self.alpha = np.ones( [N*kappa,1] ) / (N*kappa)
			self.Y = ( ( .5 + (self.X.reshape(N,1,self.d) > np.transpose(self.X.reshape(N,1,self.d),[1,0,2])).prod(2).sum(1,dtype=float) ) / N ).reshape([N,1])
			self.Gamma = np.repeat(self.gamma,N).reshape([N*kappa,1])
			counter = 0

			# Test stopping condition
			while not counter or not self._test_stop():
				
				# Select working set
				(i,j) = self._select_working_set()
				
				# Solve sub-problem
				(alpha_i, alpha_j) = self._sub_problem(i,j)
				
				print self.alpha[i] - alpha_i
				print self.alpha[j] - alpha_j
				
				# Update alpha
				self.alpha[i] = alpha_i
				self.alpha[j] = alpha_j
				
				counter += 1
			
			beta = ma.masked_less( self.alpha, 1e-8 )
			mask = ma.getmask(beta)
			self.NSV = beta.count()
			
			self.beta = self.alpha
			self.SV = np.vstack( [self.X,]*kappa )
			
			'''
			self.beta = beta.compressed().reshape([self.NSV,1])
			self.SV = ma.array( np.tile(self.X.T,kappa).T, mask=np.repeat(mask,self.d)).compressed().reshape([self.NSV,self.d])
			self.Gamma = ma.array( self.Gamma, mask=mask ).compressed().reshape([self.NSV,1])
			'''
			duration = datetime.datetime.now() - start
			print "optimized in %ss" % ( duration.seconds + float(duration.microseconds)/1000000)
		
def run():
	fig = plt.figure()
	
	samples = np.vstack( [ np.random.multivariate_normal( mean=np.array([3,3]), cov=np.array( np.identity(2) ), size=np.array([50,]) ),
		np.random.multivariate_normal( mean=np.array([7,7]), cov=np.array( np.identity(2) ), size=np.array([50,]) ) 
	] )
	
	'''
	C = arange(1e-6,1e-0,1e-1) 
	res = [ svm( samples, Lambda=c, gamma=[.125,.5,2.,8.,32.]) for c in C ]
	
	a = fig.add_subplot(1,2,1)
	a.plot( C, [mod.cdf_loss().sum() for mod in res],'r' )
	a.set_title("Loss vs Lambda")
	a.grid(True)
	
	b = fig.add_subplot(1,2,2)
	b.plot( C, [mod.NSV for mod in res] )
	b.set_title("NSV vs Lambda")
	b.grid(True)
	
	c = fig.add_subplot(2,2,3)
	c.plot( [mod.NSV for mod in res], [mod.cdf_loss().sum() for mod in res], 'o--' )
	c.set_title("NSV vs Loss")
	
	
	plt.show()
	return True
	'''
	
	#mod = svm( samples, Lambda=.005, gamma=[.125,.25,.5,1,2,4,8,16] )
	mod = svm( samples, Lambda=.005, gamma=[.125,4] )
	
	print mod
	
	X = dstack(mgrid[0:10:.1,0:10:.1]).reshape([10000,2])
	
	plt.contourf(arange(0,10,.1),arange(0,10,.1),mod.pdf(X).reshape([100,100]).T,200, antialiased=True, cmap=cm.gray )
	CS = plt.contour(arange(0,10,.1),arange(0,10,.1),mod.pdf(X).reshape([100,100]).T, [.01,], colors='r' )
	plt.plot( hsplit(samples,2)[0],hsplit(samples,2)[1], 'r+' )
	plt.scatter( hsplit(mod.SV,2)[0].reshape([mod.NSV,]),hsplit(mod.SV,2)[1].reshape([mod.NSV],), s=(mod.NSV*200*mod.beta.reshape([mod.NSV,])), alpha=.25, color='r' )
	plt.clabel(CS, inline=1, fontsize=10)
	
	plt.axis( [0,10,0,10] )
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
