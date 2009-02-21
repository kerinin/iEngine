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
import cvxmod
from cvxopt import *

from numpy import *

import matplotlib.pyplot as plt


cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['abstol'] = 1e-15
cvxopt.solvers.options['reltol'] = 1e-15
cvxopt.solvers.options['feastol'] = 1e-15

_Functions = ['run']
	
class svm:
	def __init__(self,data=list(),C=.1, gamma =arange(1,4,1) ):
		self.X = data
		self.N = len(data)
		
		self.C = C
		self.gamma = gamma
		
		self.Gamma = None
		self.Y = None
		self.SV = None
		self.NSV = None
		self.alpha = None
		self.beta = None
		self.d = None
		self.K = None
		
		self._compute()
		
	def __str__(self):
		ret = "SVM Instance\n"
		ret += "X: (%s x %sd)\n" % (self.N, self.d)
		ret += "C: %s\n" % self.C
		ret += "gamma: %s\n" % str(self.gamma)
		ret += "SV: %s (%s percent)\n" % ( self.NSV,100. * float(self.NSV) / float(self.N ) )
		ret += "Loss: %s\n" % self.cdf_res().sum()
		return ret
	
	def _Omega(self,Gamma):
		return 1e0+(Gamma ** self.C)
		
	def _K(self,X,Y,gamma):
		N = X.size
		M = Y.size
		
		# Unbiased
		# return ones([N,M])/10.
		
		# Sigmoid
		return ( 1.0 / ( 1.0 + numpy.exp( -gamma * (X-Y) ) ) ).reshape(N,M)
		
		# RBF
		#return ( exp( -((X-Y)**2.0) / gamma ) ).reshape(N,M)

	def _k(self,X,Y,gamma):
		diff = X-Y
		N = X.size
		M = Y.size
		
		return ( gamma / ( 2.0 + numpy.exp( gamma * diff ) + numpy.exp( -gamma * diff ) ) ).reshape(N,M)
		
	def cdf(self,x):
		#print 'cdf: %s' % repr(x.shape)
		return numpy.dot( self._K( atleast_2d(x).T, self.SV, self.Gamma ), self.beta.T )
		
	def pdf(self,x):
		#print 'pdf: %s' % repr(x.shape)
		return numpy.dot( self._k( atleast_2d(x).T, self.SV, self.Gamma ), self.beta.T )
		
	def cdf_res(self,X=None):
		if X==None:
			X = self.X
		return ( self.Y.flatten() - self.cdf( X.flatten() ).flatten() )**2
	
	def __iadd__(self, points):
		# overloaded '+=', used for adding a vector list to the module's data
		# 
		# @param points		A LIST of observation vectors (not a single ovservation)
		self.X += points
	
	def _compute(self):
		start = datetime.datetime.now()

		# NOTE: From (1), the optimization problem should be:
		# min ( \sum_{i=1}^\ell ( y_i - \sum_{j=1}^\ell \sum_{n=1}^k \alpha_j^n k_n(x_i,x_j) )^2 + \lambda \sum_{i=1}^\ell \sum_{n=1}^k \frac{1}{\gamma_n} \alpha_i^n )
		# sjt \alpha_i \ge 0, i = 1,...,\ell
		
		# Gameplan: implement this minimization problem - if it works figure out what the matrix
		# definitions will be for the optimization problem and re-implement it in CVXOPT.  From
		# there you can start working on decomposition.
		
		kappa = len( self.gamma )
		(N,self.d) = self.X.shape
		self.Y = ( ( .5 + (self.X.reshape(N,1,self.d) > transpose(self.X.reshape(N,1,self.d),[1,0,2])).prod(2).sum(1,dtype=float) ) / N ).reshape([N,])
		Z = numpy.zeros([N,N])
		self.K = numpy.array( vstack(
			[ 
				numpy.hstack( ( [Z,] * i ) + [self._K( self.X.reshape([N,1]), self.X.reshape([1,N]), self.gamma[i] ),] + ( [Z,] * ( kappa-1-i ) ) )
				for i in range( kappa ) 
			]
		) )

		self.Gamma = numpy.hstack( [ numpy.tile(g,N) for g in self.gamma ] )
		
		P = cvxopt.matrix( numpy.dot(self.K.T,self.K), (N*kappa,N*kappa) )
		q = cvxopt.matrix( ( self._Omega(self.Gamma) - ( numpy.ma.dot( tile(self.Y,kappa), self.K ) ) ), (N*kappa,1) )
		
		G = cvxopt.matrix( -identity(N*kappa), (N*kappa,N*kappa) )
		h = cvxopt.matrix( 0.0, (N*kappa,1) )
		A = cvxopt.matrix( 1., (1,N*kappa) )
		b = cvxopt.matrix( 1., (1,1) )
		#print "P: %s, q: %s, G: %s, h: %s, A: %s, b: %s" % (P.size,q.size,G.size,h.size,A.size,b.size)

		# Solve!
		p = solvers.qp( P=P, q=q, G=G, h=h, A=A, b=b )
		
		beta = ma.masked_less( p['x'], 1e-8 )
		mask = ma.getmask(beta)
		self.alpha = beta
		self.beta = numpy.atleast_2d( beta.compressed() )
		self.SV = numpy.atleast_2d( numpy.ma.array( numpy.tile(self.X.T,kappa).T, mask=mask).compressed() )
		self.Gamma = numpy.atleast_2d( numpy.ma.array( self.Gamma, mask=mask ).compressed() )
		self.NSV = self.beta.size
		
		duration = datetime.datetime.now() - start
		print "optimized in %ss" % ( duration.seconds + float(duration.microseconds)/1000000)
		
def run():
	#samples = array([[gauss(0,1)] for i in range(50) ] + [[gauss(8,1)] for i in range(50) ]).reshape([100,1]) 
	samples = array([[gauss(0,1)] for i in range(40) ] ).reshape([40,1]) 
	#samples = arange(0,4).reshape([4,1])
	#samples = array( [ [i,i+.1,i+.2] for i in range(0,10) ], dtype=float ).reshape([30,1])
	#samples = list()

	start = -5.
	end = 12.
	X = arange(start,end,.25)
	fig = plt.figure()
	
	#C = [-1e1,-1e0,-1e-1,-1e-2,-1e-3,-1e-4,-1e-5,0.,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1]
	#C = [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7]
	C = arange(-10,0,1) 
	'''
	res = [ svm( samples, C=c, gamma=[.5,1.,2.,4.,8.]) for c in C ]
	
	a = fig.add_subplot(1,2,1)
	a.plot(numpy.sort( res[0].X,0), numpy.sort( res[0].Y,0), 'green' )
	for mod in res:
		a.plot(X, mod.cdf(X), '--' )
	a.set_title("Computed vs emprical CDF")
	
	b = fig.add_subplot(1,2,2)
	b.plot( [mod.NSV for mod in res], [mod.cdf_res().sum() for mod in res], 'o--')
	b.set_title("NSV vs Loss")
	
	plt.show()
	return True
	'''
	
	# 2 good, 10 bad
	# 5->4
	# 4->4
	# 3->3
	
	mod = svm( numpy.sort(samples),C=-.5, gamma=[.01,.1,1,10,100] )
	#mod = svm( numpy.sort(samples),C=math.exp(1), gamma=[3.,] )
	
	print mod
	
	a = fig.add_subplot(2,2,1)
	#a.hist(mod.K.compressed().flatten(), 20, normed=1)
	#a.set_title("K distribution")
	#a.plot( [ i % mod.N for i in range( mod.N * len(mod.gamma) ) ], mod.alpha, 'o' )
	#a.set_title("weights (x=ell)")
	a.plot(mod.Gamma, mod.beta,  'o')
	a.set_title('gamma vs weight')
	
	b = fig.add_subplot(2,2,3)
	n, bins, patches = b.hist(mod.X, 20, normed=1, facecolor='green', alpha=0.5, label='empirical distribution')
	b.plot(X,mod.pdf(X), 'r--', label="computed distribution")
	b.set_title("Computed vs empirical PDF")
	
	c = fig.add_subplot(2,2,2)
	c.plot(numpy.sort(mod.X,0), numpy.sort(mod.Y,0), 'green' )
	c.plot(X, mod.cdf(X), 'r--' )
	c.plot( mod.X, mod.cdf_res(mod.X), '+' )
	
	#c.plot( mod.X, (mod.Y.reshape( [len(mod.X),]) - mod.cdf( mod.X.reshape( [len(mod.X),]) ) ) ** 2, '+' )
	c.set_title("Computed vs emprical CDF")
	
	d = fig.add_subplot(2,2,4)
	for i in range( mod.NSV ):
		d.plot( X, numpy.dot( mod._K( atleast_2d(X).T, mod.SV[0][i], mod.Gamma[0][i] ), mod.beta[0][i].T ) )
		#d.plot( X, numpy.dot( mod._K( X, mod.SV[i], mod.gamma[i] ), mod.beta[i] ) )
	d.plot( mod.SV, mod.beta/2, 'o' )
	d.set_title("SV Contributions")
	
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
