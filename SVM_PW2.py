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
	
class subset:
	def __init__(self,X,D,tStart,theta):
		self.N,self.d = X.shape
		self.X = X
		self.D = D
		self.argStart = np.argmax( X * ( X == tStart ) )
		self.argEnd = np.argmax( X * (X < (tStart + theta) ) ) + 1
		print self.argEnd - self.argStart
		
	def __sub__(self,S):
		
		phi = self.D[self.argStart:self.argEnd,S.argStart:S.argEnd].sum(1) / S.N
		
		diff = ( phi * np.log2(phi) ).sum()
		#print self.D[self.argStart:self.argEnd,S.argStart:S.argEnd].shape
		return diff

class svm:
	def __init__(self,data=list(),Lambda=.1, gamma =.5, theta=None ):
	# SVM Class
	#
	# @param data		[Nxd] array of observations where N is the number of observations and d is the dimensionality of the abstract space
	# @param Lambda		Regularizer to control Smoothness / Accuracy.  Preliminary experimental results show the range 0-1 controls this parameter.
	# @param gamma		List of gamma values which define the kernel smoothness
	
		try:
			self.N,self.d = data.shape
		except ValueError:
			self.N,self.d = (len(data),1)
			self.X = data.reshape([ self.N, self.d ])
		else:
			self.X = data

		self.Lambda = Lambda
		self.gamma = gamma
		
		self.t = np.hsplit(self.X,[1])[0]
		self.offset = np.tile( np.hsplit(self.X,[1])[0], len(theta) )
		self.theta = np.repeat( np.array(theta), self.N )
		
		self.D = self._K( self.X.reshape([self.N,1,self.d]) - self.X.T.reshape([1,self.N,self.d]) )
		self.S = np.array( [ [ subset(self.X,self.D, t, theta ) for t in self.t ] for theta in self.theta ] ).flatten()
		
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
		
	def _K(self,X,Y=None):
	# Kernel function
		
		# Subset difference
		if not Y==None:
			diff = ( X - Y )
		else:
			diff =X

		
		# Subset Gaussian
		return (1.0/(self.gamma*math.sqrt(math.pi))) * np.ma.exp( (-1.*(np.ma.power(diff,2)))/self.gamma)
		
	def pdf(self,Sx,X):
	# Probability distribution function
	#
		pass
		
	def _compute(self):
		start = datetime.datetime.now()

		diff = np.array( [ [ Si - Sj for Sj in self.S ] for Si in self.S ] )

		self.K = self._K( diff )
		
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
		self.beta = beta
		#self.beta = beta.compressed().reshape([self.NSV,1])
		#self.SV = np.array( [ subsetSV( S=S,svm=self ) for S in np.ma.array( self.S, mask=mask).compressed() ],ndmin=2)

		duration = datetime.datetime.now() - start
		print "optimized in %ss" % ( duration.seconds + float(duration.microseconds)/1000000)
		
	def contourPlot(self, S, fig, xrange, yrange, xstep, ystep, axes=(0,1) ):
		N,d=S.shape
		xN = int((xrange[1]-xrange[0])/xstep)
		yN =  int((yrange[1]-yrange[0])/ystep)
		X = np.dstack(np.mgrid[xrange[0]:xrange[1]:xstep,yrange[0]:yrange[1]:ystep]).reshape([ xN * yN,2])
		x = np.arange(xrange[0],xrange[1],xstep)
		y = np.arange(yrange[0],yrange[1],ystep)
		
		CS1 = fig.contourf(x,y,self.pdf(S,X).reshape([xN,yN]).T,200, antialiased=True, cmap=cm.gray )
		#CS2 = plt.contour(x,y,self.pdf(S,X).reshape([xN,yN]).T, [.1,], colors='r' )
		fig.plot( np.hsplit( S,d )[0],np.hsplit( S,d )[ axes[1] ], 'r+' )
		fig.axis( [ xrange[0],xrange[1],yrange[0],yrange[1] ] )
		#return (CS1,CS2)
		
def run():
	fig = plt.figure()
	
	Xtrain = np.arange(0,10,.5)
	Ytrain = np.sin(Xtrain) + (np.random.randn( Xtrain.shape[0] )/10.)
	mod = svm( data=np.hstack([Xtrain.reshape([Xtrain.shape[0],1]),Ytrain.reshape([Ytrain.shape[0],1])]), gamma=.005, Lambda=.05, theta=[.1] )
	print mod

	Xtest = np.arange(5,11,.2)
	Ytest = np.sin(Xtest)+ (np.random.randn( Xtest.shape[0] )/10.)
	S=np.hstack([Xtest.reshape([Xtest.shape[0],1]),Ytest.reshape([Ytest.shape[0],1])])
	
	mod.contourPlot( S, plt, (0,10), (-2,2),1.,.25 )
	#(c1,c2) = mod.contourPlot( S, plt, (0,10), (-2,2),1.,.25 )

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
