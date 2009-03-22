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
		
		# Subset difference
		if not Y==None:
			diff = ( X - Y )
		else:
			diff =X

		
		# Subset Gaussian
		return (1.0/(self.gamma*math.sqrt(math.pi))) * np.ma.exp( (-1.*(np.ma.power(diff,2)))/self.gamma)
	
class subset:
	def __init__(self,data,D,svm=None,tStart=None,theta=None):
		
		self.tStart = tStart
		self.theta = theta
		self.X = data
		self.N,self.d = data.shape
		self.t = np.hsplit(self.X,[1,])[0]
		self.D = D
		self.svm = svm
		
		if tStart != None and theta != None:
			self.argStart = np.ma.masked_less(self.t,tStart,copy=False).argmin()
			self.argEnd = 1+ np.ma.masked_greater(self.t,tStart+theta,copy=False).argmax()
			
			self.N = self.argStart - self.argEnd
		else:
			self.argStart = 0
			self.argEnd = -1
			
			self.N = self.X.shape[0]
			
	def __sub__(self,other):
	# difference - for comparing two subsets using Entropy
		
		#NOTE: change this to conform to the entropy kernel
		
		if other.__class__ == subset:
		# other is an array of subsets
			Dx = self.D[self.argStart:self.argEnd,other.argStart:other.argEnd,1:].prod(2)
			Dt = self.svm._K( (self.t[self.argStart:self.argEnd] - self.t[self.argStart] ) - (other.t[other.argStart:other.argEnd] - other.t[other.argStart] ).T )
			D = ( Dx * Dt )/( self.N * other.N )
			return -( D * np.log2(D) ).sum()
			#NOTE: when comparing a point to itself, the kernel value is 0, which produces NaN in Log
		
		raise StandardError, 'This type of subtraction not implemented'

class svm(kMachine):
	def __init__(self,t=list(),data=list(),Lambda=.1, gamma =.5, theta=None ):
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
		self.t = np.hsplit(self.X,[1])[0]
		
		self.theta = theta
		self.Lambda = Lambda
		super(svm, self).__init__(gamma)
		
		self.D = self._K( self.X.reshape([self.N,1,self.d]) - self.X.T.reshape([1,self.N,self.d]) )
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
					[ subset(data=self.X,D=self.D,svm=self,tStart=tStart,theta=theta) for tStart in self.t ] 
				) for theta in self.theta 
			]
		)
		
		return S
		
	def pdf(self,S,X):
	# Probability distribution function
	#
		Ns,ds = S.shape
		Nx,dx = X.shape
		
		shift = np.array([S[0,0],]+([0,]*(ds-1))).reshape([1,ds])
		floor = np.array([0,]+([1,]*(ds-1))).reshape([1,ds])
		
		Sx = S - np.repeat( shift, Ns, axis=0 )
		Xshifted = X
		
		K = np.array([
			self._K( Sx.reshape([Ns,1,ds]) - S[0].X[S[0].argStart:S[0].argEnd].T.reshape([1,(S[0].argEnd - S[0].argStart),S[0].d]) ).prod(2).sum() / (Ns * (S[0].argEnd - S[0].argStart ) )
			for S in self.S
		])
		Ds = -( K * np.log2(K) ).reshape([self.S.shape[0],1])
		
		tmp = list()
		for S in self.S:
			shape1 = [ X.shape[0],1,X.shape[1] ]
			shape2 = [1,( S[0].argEnd-S[0].argStart),S[0].d]
			phi = self._K( Xshifted.reshape(shape1) - S[0].X[S[0].argStart:S[0].argEnd].reshape(shape2 ) ).prod(2).sum(1) / ( Nx * (S[0].argEnd-S[0].argStart) ) 
			tmp.append( -( phi * np.log2(phi) ) )
		Dx = np.array(tmp)
				
		R = Ds + Dx
		
		pdf = np.ma.dot( R.T, self.beta )
		
		return pdf
	
	def _compute(self):
		start = datetime.datetime.now()
		
		diff = np.array( self.S - self.S.T, dtype=float, copy=False )
		
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

		duration = datetime.datetime.now() - start
		print "optimized in %ss" % ( duration.seconds + float(duration.microseconds)/1000000)
		
	def contourPlot(self, S, fig, xrange, yrange, xstep, ystep, axes=(0,1) ):
		xN = int((xrange[1]-xrange[0])/xstep)
		yN =  int((yrange[1]-yrange[0])/ystep)
		X = np.dstack(np.mgrid[xrange[0]:xrange[1]:xstep,yrange[0]:yrange[1]:ystep]).reshape([ xN * yN,2])
		x = np.arange(xrange[0],xrange[1],xstep)
		y = np.arange(yrange[0],yrange[1],ystep)

		CS1 = fig.contourf(x,y,self.pdf(S,X).reshape([xN,yN]).T,200, antialiased=True, cmap=cm.gray )
		#CS2 = plt.contour(x,y,self.pdf(S,X).reshape([xN,yN]).T, [.1,], colors='r' )
		fig.plot( np.hsplit( S,S.shape[1] )[0],np.hsplit( S,S.shape[1] )[ axes[1] ], 'r+' )
		#fig.axis( [ xrange[0],xrange[1],yrange[0],yrange[1] ] )
		#return (CS1,CS2)
		
def run():
	fig = plt.figure()
	
	Xtrain = np.arange(0,20,.25)
	Ytrain = np.sin(Xtrain) + (np.random.randn( Xtrain.shape[0] )/10.)
	mod = svm( data=np.hstack([Xtrain.reshape([Xtrain.shape[0],1]),Ytrain.reshape([Ytrain.shape[0],1])]), gamma=.5, Lambda=.005, theta=[5.] )
	print mod

	Xtest = np.arange(5,15,.25)
	Ytest = np.sin(Xtest)+ (np.random.randn( Xtest.shape[0] )/10.)
	S=np.hstack([Xtest.reshape([Xtest.shape[0],1]),Ytest.reshape([Ytest.shape[0],1])])
	
	mod.contourPlot( S, plt, (0,20), (-2,2),1.,.1 )
	#(c1,c2) = mod.contourPlot( S, plt, (0,20), (-2,2),.1,.01 )

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
