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
cvxopt.solvers.options['abstol'] = 1e-10

_Functions = ['run']
	
class svm:
	def __init__(self,data=list(),C=1e-1, gamma =[.1,] ):
		self.X = data
		
		self.C = C
		self.gamma = gamma
		
		self.Y = None
		self.SV = None
		self.betas = None
		self.d = None
		self.K = None
		
		self._compute()
		
	def __str__(self):
		ret = "SVM Instance\n"
		ret += "X: (%s x %sd)\n" % (len(self.X), self.d)
		ret += "C: %s\n" % self.C
		ret += "gamma: %s\n" % str(self.gamma)
		ret += "SV: %s (%s percent)\n" % ( len(self.beta),100. * float(len(self.SV)) / float(len(self.X) ) )
		ret += "Loss: %s\n" % self.cdf_res().sum()
		return ret
	
	def _Omega(self,Gamma):
		return self.C / Gamma
		
	def _K(self,X,Y,gamma):
		N = X.size
		M = Y.size
		
		# Sigmoid
		return ( 1.0 / ( 1.0 + numpy.exp( -gamma * (X-Y) ) ) ).reshape(N,M)
		
		# RBF
		# return ( exp( -((X-Y)**2.0) / gamma ) ).reshape(N,M)

	def _k(self,X,Y,gamma):
		diff = X-Y
		N = X.size
		M = Y.size
		
		return ( gamma / ( 2.0 + numpy.exp( gamma * diff ) + numpy.exp( -gamma * diff ) ) ).reshape(N,M)
		
	def cdf(self,x):
		#print 'cdf: %s' % repr(x.shape)
		return numpy.dot( self._K( atleast_2d(x).T, self.SV, self.gamma ), self.beta.T )
		
	def pdf(self,x):
		#print 'pdf: %s' % repr(x.shape)
		return numpy.dot( self._k( atleast_2d(x).T, self.SV, self.gamma ), self.beta.T )
		
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
		self.Y = ( (self.X.reshape(N,1,self.d) > transpose(self.X.reshape(N,1,self.d),[1,0,2])).prod(2).sum(1,dtype=float) / N ).reshape([N,])
		Z = numpy.zeros([N,N])
		self.K = numpy.ma.masked_less( vstack(
			[ 
				numpy.hstack( ( [Z,] * i ) + [self._K( self.Y.reshape(N,1,self.d), transpose(self.Y.reshape(N,1,self.d), [1,0,2]), self.gamma[i] ),] + ( [Z,]*(kappa-i-1 ) ) )
				for i in range( kappa ) 
			]
		), 1e-10 )
		Gamma = numpy.hstack( [ numpy.tile(g,N) for g in self.gamma ] )
			
		P = cvxopt.matrix( numpy.dot(self.K.T,self.K), (N*kappa,N*kappa) )
		q = cvxopt.matrix( ( self._Omega(Gamma) - ( 2.0 * numpy.ma.dot( tile(self.Y,kappa), self.K ) ) ), (N*kappa,1) )
		G = cvxopt.matrix( -identity(N*kappa), (N*kappa,N*kappa) )
		h = cvxopt.matrix( 0.0, (N*kappa,1) )
		A = cvxopt.matrix( 1., (1,N*kappa) )
		b = cvxopt.matrix( 1., (1,1) )
		#print "P: %s, q: %s, G: %s, h: %s, A: %s, b: %s" % (P.size,q.size,G.size,h.size,A.size,b.size)
		
		# Solve!
		p = solvers.qp( P=P, q=q, G=G, h=h, A=A, b=b )
		
		mask = ma.make_mask( array(p['x']) < 1e-5 )
		self.beta = numpy.atleast_2d( numpy.ma.array( array(p['x']), mask=mask ).compressed() )
		self.SV = numpy.atleast_2d( numpy.ma.array( numpy.tile(self.X,kappa), mask=mask).compressed() )
		self.gamma = numpy.atleast_2d( numpy.ma.array( Gamma, mask=mask ).compressed() )
		
		duration = datetime.datetime.now() - start
		print "optimized in %ss" % (float(duration.microseconds)/1000000)
		
def run():
	#samples = array([[gauss(0,1)] for i in range(20) ] + [[gauss(8,1)] for i in range(20) ]).reshape([40,1]) 
	samples = array([[gauss(0,1)] for i in range(40) ] ).reshape([40,1]) 
	C = [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2]
	
	#res = [ svm( samples, C=c ) for c in C ]
	
	#plt.plot( C, [ m.cdf_res().sum() for m in res ], 'o--' )
	#plt.show()
	
	#return True
	
	mod = svm( samples,C=1e-1 )
	print mod
	
	fig = plt.figure()
	
	start = -5.
	end = 12.
	X = arange(start,end,.25)
	
	a = fig.add_subplot(2,2,1)
	#n, bins, patches = a.hist(mod.data, 20, normed=1, facecolor='green', alpha=0.5, label='empirical distribution')
	#a.plot(X,mod.pdf(X), 'r--', label="computed distribution")
	#a.set_title("Computed vs empirical PDF")
	
	a.hist(mod.K.compressed().flatten(), 20, normed=1)
	a.set_title("K distribution")
	
	b = fig.add_subplot(2,2,3)
	b.plot(mod.gamma, mod.beta,  'o')
	b.set_title('gamma vs weight')
	
	c = fig.add_subplot(2,2,2)
	c.plot(numpy.sort(mod.X,0), numpy.sort(mod.Y,0), 'green' )
	c.plot(X, mod.cdf(X), 'r--' )
	c.plot( mod.X, mod.cdf_res(mod.X), '+' )
	
	#c.plot( mod.X, (mod.Y.reshape( [len(mod.X),]) - mod.cdf( mod.X.reshape( [len(mod.X),]) ) ) ** 2, '+' )
	c.set_title("Computed vs emprical CDF")
	
	d = fig.add_subplot(2,2,4)
	for i in range( len(mod.beta) ):
		d.plot( X, numpy.dot( mod._K( atleast_2d(X).T, mod.SV[i], mod.gamma[i] ), mod.beta[i].T ) )
		#d.plot( X, numpy.dot( mod._K( X, mod.SV[i], mod.gamma[i] ), mod.beta[i] ) )
	d.set_title("SV Contributions")
	
	#plt.show()
	
	
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
