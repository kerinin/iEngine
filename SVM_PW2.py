#! /usr/bin/env python

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
import matplotlib.cm as cm



cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['abstol'] = 1e-15
cvxopt.solvers.options['reltol'] = 1e-15
cvxopt.solvers.options['feastol'] = 1e-15

_Functions = ['run']
	
class svm:
	def __init__(self,data=list(),Lambda=.1, gamma =.5 ):
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
		
	def _K(self,X,Y):
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
		return ( 1.0 / ( 1.0 + numpy.exp( -self.gamma * diff ) ) ).prod(2).reshape(N,M)
		
		# RBF
		#return ( exp( -((X-Y)**2.0) / gamma ) ).reshape(N,M)

	def _k(self,X,Y):
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
		
		return ( self.gamma / ( 2.0 + numpy.exp( self.gamma * diff ) + numpy.exp( -self.gamma * diff ) ) ).prod(2).reshape(N,M)
		
	def cdf(self,X):
	# Cumulative distribution function
	#
	# @param X				[Nxd] array of points for which to calculate the CDF
		
		return numpy.ma.dot( self._K( X, self.SV ), self.beta )
		
	def pdf(self,X):
	# Probability distribution function
	#
	# @param X				[Nxd] array of points for which to calculate the PDF
	
		return numpy.ma.dot( self._k( X, self.SV ), self.beta )
		
	def cdf_res(self,X=None):
	# CDF residuals
	#
		if X==None:
			X = self.X
		#return ( self.Y.flatten() - self.cdf( X.flatten() ).flatten() )
		return self.Y - self.cdf(X)
		
	def cdf_loss(self,X=None):
		
		return sqrt( (self.cdf_res(X)**2).sum(1) )
		
	def __iadd__(self, points):
		# overloaded '+=', used for adding a vector list to the module's data
		# 
		# @param points		A LIST of observation vectors (not a single ovservation)
		raise StandardError, 'Not Implemented'
		self.X += points
	
	def _compute(self):
		start = datetime.datetime.now()

		self.Y = ( ( .5 + (self.X.reshape(self.N,1,self.d) > transpose(self.X.reshape(self.N,1,self.d),[1,0,2])).prod(2).sum(1,dtype=float) ) / self.N ).reshape([self.N,1])
		self.K = self._K( self.X, self.X )
		
		P = cvxopt.matrix( numpy.dot(self.K.T,self.K), (self.N,self.N) )
		#q = cvxopt.matrix( ( self._Omega(self.Gamma) - ( numpy.ma.dot( self.K.T, self.Y ) ) ), (N*kappa,1) )
		q = cvxopt.matrix( ( self.Lambda / self.K.T.sum(0) ) - ( ( 1./self.N ) * ( numpy.dot( self.K.T, self.K ).sum(0) ) ) )
		G = cvxopt.matrix( -identity(self.N), (self.N,self.N) )
		h = cvxopt.matrix( 0.0, (self.N,1) )
		A = cvxopt.matrix( 1., (1,self.N) )
		b = cvxopt.matrix( 1., (1,1) )
		
		
		print P.size
		print q.size
		
		# Solve!
		p = solvers.qp( P=P, q=q, G=G, h=h, A=A, b=b )
		
		print p['x']
		
		beta = ma.masked_less( p['x'], 1e-8 )
		mask = ma.getmask(beta)
		self.NSV = beta.count()
		self.alpha = beta
		self.beta = beta.compressed().reshape([self.NSV,1])
		self.SV = numpy.ma.array( self.X, mask=numpy.repeat(mask,self.d)).compressed().reshape([self.NSV,self.d])
		#self.beta = beta
		#self.SV = self.X

		duration = datetime.datetime.now() - start
		print "optimized in %ss" % ( duration.seconds + float(duration.microseconds)/1000000)
		
	def contourPlot(self, fig, xrange, yrange, xstep, ystep, axes=(0,1) ):
		xN = int((xrange[1]-xrange[0])/xstep)
		yN =  int((yrange[1]-yrange[0])/ystep)
		X = dstack(mgrid[xrange[0]:xrange[1]:xstep,yrange[0]:yrange[1]:ystep]).reshape([ xN *yN,2])
		x = arange(xrange[0],xrange[1],xstep)
		y = arange(yrange[0],yrange[1],ystep)

		CS1 = fig.contourf(x,y,self.pdf(X).reshape([xN,yN]).T,200, antialiased=True, cmap=cm.gray )
		CS2 = plt.contour(x,y,self.pdf(X).reshape([xN,yN]).T, [.1,], colors='r' )
		fig.plot( hsplit( self.X,self.d )[ axes[0] ],hsplit( self.X,self.d )[ axes[1] ], 'r+' )
		fig.scatter( hsplit(self.SV,self.d)[ axes[0] ].reshape([self.NSV,]),hsplit(self.SV,self.d)[ axes[1] ].reshape([self.NSV],), s=(self.NSV*200*self.beta.reshape([self.NSV,])), alpha=.25, color='r' )
		#fig.clabel(CS, inline=1, fontsize=10)
		fig.axis( [ xrange[0],xrange[1],yrange[0],yrange[1] ] )
		return (CS1,CS2)
		
def run():
	fig = plt.figure()
	
	Xtrain = numpy.arange(0,20,.2)
	Ytrain = numpy.sin(Xtrain) + (numpy.random.randn( Xtrain.shape[0] )/10.)
	
	Xtest = numpy.arange(5,15,.1)
	Ytest = numpy.sin(Xtest)+ (numpy.random.randn( Xtest.shape[0] )/10.)
	
	mod = svm( numpy.vstack([Xtrain,Ytrain]).T, gamma=5., Lambda=.0005 )

	#plt.plot(hsplit(samples,2)[0], hsplit(samples,2)[1], 'o')
	(c1,c2) = mod.contourPlot( plt, (0,20), (-2,2),.1,.01 )

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
