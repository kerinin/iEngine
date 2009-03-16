#! /usr/bin/env python

import sys, getopt, math, datetime, os, cmath
from random import gauss

import numpy as np
import scipy
import scipy.special
import scipy.stats
import cvxopt
#import cvxmod
from cvxopt import *

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm



cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['abstol'] = 1e-15
cvxopt.solvers.options['reltol'] = 1e-15
cvxopt.solvers.options['feastol'] = 1e-15

_Functions = ['run']
	
class kMachine(object):
	def __init__(self,gamma):
		self.gamma = gamma
		
	def _K(self,X,Y=None):
	# Kernel function
	#
	# @param X				[Nxd] array of observations
	# @param Y				[Mxd] array of observations
	# @param gamma			kernel width parameter
	
		#N,d1 = X.shape
		#M,d2 = Y.shape
		
		#if d1 != self.d != d2:
		#	raise StandardError, 'Matrices do not conform to the dimensionality of existing observations'
		
		#diff = X.reshape([N,1,self.d]) - np.transpose( Y.reshape([M,1,self.d]), [1,0,2] )
		
		# Subset difference
		if not Y==None:
			diff = np.asfarray( X - Y )
		else:
			diff = np.asfarray( X )
		
		# Gaussian
		#return (1.0/(self.gamma*math.sqrt(math.pi))) * np.exp( (diff**2)/-self.gamma).prod(2).reshape(N,M)
		
		# Subset Gaussian
		return (1.0/(self.gamma*math.sqrt(math.pi))) * np.exp( (diff**2)/-self.gamma)
		
		# Sigmoid
		#return ( 1.0 / ( 1.0 + np.exp( -self.gamma * diff ) ) ).prod(2).reshape(N,M)
	
class subset(kMachine):
	def __init__(self,t,data,gamma,tStart=None,theta=None):

		try:
			self.N,self.d = data.shape
		except ValueError:
			self.N,self.d = (len(data),1)
			
		self.xTest = None
		
		self.tStart = tStart
		self.theta = theta
		super(subset, self).__init__(gamma)
		
		if tStart and theta:
			mask = np.ma.mask_or( (self.tStart < t), ( t >= (self.tStart+self.theta) ) )
			self.t = np.ma.array(data,mask=mask)
			self.x = np.ma.array(data,mask=mask)
		else:
			self.t = np.ma.array(t)
			self.x = np.ma.array(data)
		
		# Why is t [1xn]?  this seems stupid - just make t a column vector and you eliminate the .T down there
		
		self.X = np.vstack( [self.t.compressed(), self.x.compressed()] ).T
		
	def __sub__(self,other):
	# difference - for comparing two subsets using symmetric KL divergence
		
		#if other.__class__ == np.ndarray or other.__class__ == np.ma.core.MaskedArray:
		if other.__class__ == subset:
		# other is an array of subsets
			X = np.vstack( [self.X, other.X] )
			
			pSelf = self._Pr( X )
			pOther = other._Pr( X )
			logpSelf = np.log2(pSelf)
			logpOther = np.log2(pOther)
			
			return  ( pSelf * logpSelf ).sum() + ( pOther * logpOther ).sum() - ( pSelf * logpOther ).sum() - ( pOther * logpSelf ).sum()
		if other.__class__ == np.ndarray or other.__class__ == np.ma.core.MaskedArray:
			if other.dtype == float:
			# other is an array of floats
				X = np.vstack( [self.X, other] )
				
				pSelf = self._Pr( X )
				pOther = other._Pr( X,other )
				logpSelf = np.log2(pSelf)
				logpOther = np.log2(pOther)
				
				return  ( pSelf * logpSelf ).sum() + ( pOther * logpOther ).sum() - ( pSelf * logpOther ).sum() - ( pOther * logpSelf ).sum()		
			
		raise StandardError, 'This type of subtraction not implemented'

	
	def _Pr(self,X,Y=None):
		if not Y:
			Y = self.X
		N,d1 = X.shape
		M,d2 = Y.shape
		
		# PROBLEM: This isn't accounting for t
		if N and M:
			sum =  self._K(
				X.reshape([N,1,d1]), np.transpose( Y.reshape([M,1,d2]), [1,0,2] ) 
			).prod(2).reshape(N,M).sum(1)
			
			return ( 1./N ) * sum
		else:
			return 0.

class svm(kMachine):
	def __init__(self,t=list(),data=list(),Lambda=.1, gamma =.5, theta=None ):
	# SVM Class
	#
	# @param data		[Nxd] array of observations where N is the number of observations and d is the dimensionality of the abstract space
	# @param Lambda		Regularizer to control Smoothness / Accuracy.  Preliminary experimental results show the range 0-1 controls this parameter.
	# @param gamma		List of gamma values which define the kernel smoothness
	
		try:
			self.t = t
			self.N,self.d = data.shape
		except ValueError:
			self.t = t
			self.N,self.d = (len(self.X),1)
			self.X = data.reshape([ self.N, self.d ])
		else:
			self.t = t
			self.X = data
		
		self.theta = theta
		self.Lambda = Lambda
		super(svm, self).__init__(gamma)
		
		self.S = self._S()
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
		return ret
		
	def _S(self):
		
		S = np.ma.vstack( 
			[ 
				np.ma.vstack( 
					[ subset(t=self.t,data=self.X,tStart=tStart,theta=theta,gamma=self.gamma) for tStart in self.t ] 
				) for theta in self.theta 
			]
		)
		
		return S
		
	def pdf(self,S,x):
	# Probability distribution function
	#
	# @param X				Set of training observations
	# @param X				[Nxd] array of points for which to calculate the PDF
		
		diffS = np.array([S,]) - self.SV.T

		Sx = np.vstack( [ subset( t=np.split(row,[1])[0], data = np.split(row,[1])[1], gamma=self.gamma) for row in x ] )

		diffX = Sx - self.SV.T
		
		return np.ma.dot( self._K( diffS + diffX ), self.beta )
		
	def __iadd__(self, points):
		# overloaded '+=', used for adding a vector list to the module's data
		# 
		# @param points		A LIST of observation vectors (not a single ovservation)
		raise StandardError, 'Not Implemented'
		self.X += points
	
	def _compute(self):
		start = datetime.datetime.now()

		self.K = self._K( self.S, self.S.T )
		
		P = cvxopt.matrix( np.dot(self.K.T,self.K), (self.N,self.N) )
		q = cvxopt.matrix( ( self.Lambda / self.K.T.sum(0) ) - ( ( 1./self.N ) * ( np.dot( self.K.T, self.K ).sum(0) ) ) )
		G = cvxopt.matrix( -np.identity(self.N), (self.N,self.N) )
		h = cvxopt.matrix( 0.0, (self.N,1) )
		A = cvxopt.matrix( 1., (1,self.N) )
		b = cvxopt.matrix( 1., (1,1) )
		
		# Solve!
		p = solvers.qp( P=P, q=q, G=G, h=h, A=A, b=b )
		
		beta = np.ma.masked_less( p['x'], 1e-8 )
		mask = np.ma.getmask(beta)
		self.NSV = beta.count()
		self.alpha = beta
		self.beta = beta.compressed().reshape([self.NSV,1])
		self.SV = np.ma.array( self.S, mask=mask).compressed().reshape([self.NSV,1])

		duration = datetime.datetime.now() - start
		print "optimized in %ss" % ( duration.seconds + float(duration.microseconds)/1000000)
		
	def contourPlot(self, S, fig, xrange, yrange, xstep, ystep, axes=(0,1) ):
		xN = int((xrange[1]-xrange[0])/xstep)
		yN =  int((yrange[1]-yrange[0])/ystep)
		X = np.dstack(np.mgrid[xrange[0]:xrange[1]:xstep,yrange[0]:yrange[1]:ystep]).reshape([ xN * yN,2])
		x = np.arange(xrange[0],xrange[1],xstep)
		y = np.arange(yrange[0],yrange[1],ystep)

		CS1 = fig.contourf(x,y,self.pdf(S,X).reshape([xN,yN]).T,200, antialiased=True, cmap=cm.gray )
		CS2 = plt.contour(x,y,self.pdf(S,X).reshape([xN,yN]).T, [.1,], colors='r' )
		fig.plot( S.t,np.hsplit( S.x,S.d )[ axes[1]-1 ], 'r+' )
		fig.axis( [ xrange[0],xrange[1],yrange[0],yrange[1] ] )
		return (CS1,CS2)
		
def run():
	fig = plt.figure()
	
	Xtrain = np.arange(0,20,2)
	Ytrain = np.sin(Xtrain) + (np.random.randn( Xtrain.shape[0] )/10.)
	mod = svm( Xtrain.reshape([Xtrain.shape[0],1]), Ytrain.reshape([Ytrain.shape[0],1]), gamma=.5, Lambda=.5, theta=[.1] )


	Xtest = np.arange(5,15,.1)
	Ytest = np.sin(Xtest)+ (np.random.randn( Xtest.shape[0] )/10.)
	S = subset(Xtest,Ytest,gamma=.5)
	
	(c1,c2) = mod.contourPlot( S, plt, (0,20), (-2,2),.1,.01 )

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
