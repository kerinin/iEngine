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

_Functions = ['run']
	
class svm:
	def __init__(self,data=list(),C=1e-1, gamma =[ (2./3.)**i for i in range(-5,5) ] ):
		self.data = data
		self.Y = None
		self.SV = None
		self.betas = None
		self.d = None
		self.K = None
		
		self.C = C
		self.gamma = gamma
		
		self._compute()
	
	def _Omega(self,Gamma):
		return 0
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
		
	def __iadd__(self, points):
		# overloaded '+=', used for adding a vector list to the module's data
		# 
		# @param points		A LIST of observation vectors (not a single ovservation)
		self.data += points
	
	def _compute(self):
		start = datetime.datetime.now()

		# NOTE: From (1), the optimization problem should be:
		# min ( \sum_{i=1}^\ell ( y_i - \sum_{j=1}^\ell \sum_{n=1}^k \alpha_j^n k_n(x_i,x_j) )^2 + \lambda \sum_{i=1}^\ell \sum_{n=1}^k \frac{1}{\gamma_n} \alpha_i^n )
		# sjt \alpha_i \ge 0, i = 1,...,\ell
		
		# Gameplan: implement this minimization problem - if it works figure out what the matrix
		# definitions will be for the optimization problem and re-implement it in CVXOPT.  From
		# there you can start working on decomposition.
		
		
		C = self.C
		gamma = self.gamma
		kappa = len( gamma )
		(N,d) = self.data.shape
		X = self.data
		Y = ( (X.reshape(N,1,d) > transpose(X.reshape(N,1,d),[1,0,2])).prod(2).sum(1,dtype=float) / N ).reshape([N,])
		Z = numpy.zeros([N,N])
		K = numpy.ma.masked_less( vstack(
			[ 
				numpy.hstack( ( [Z,] * i ) + [self._K( Y.reshape(N,1,d), transpose(Y.reshape(N,1,d), [1,0,2]), gamma[i] ),] + ( [Z,]*(kappa-i-1 ) ) )
				for i in range( kappa ) 
			]
		), 1e-10 )
		Gamma = numpy.hstack( [ numpy.tile(g,N) for g in gamma ] )
			
		P = cvxopt.matrix( numpy.dot(K.T,K), (N*kappa,N*kappa) )
		q = cvxopt.matrix( ( self._Omega(Gamma) - ( 2.0 * numpy.ma.dot( tile(Y,kappa), K ) ) ), (N*kappa,1) )
		G = cvxopt.matrix( -identity(N*kappa), (N*kappa,N*kappa) )
		h = cvxopt.matrix( 0.0, (N*kappa,1) )
		A = cvxopt.matrix( 1., (1,N*kappa) )
		b = cvxopt.matrix( 1., (1,1) )
		#print "P: %s, q: %s, G: %s, h: %s, A: %s, b: %s" % (P.size,q.size,G.size,h.size,A.size,b.size)
		
		# Solve!
		p = solvers.qp( P, q, G, h, A, b )
		
		alpha = array(p['x'])
		mask = ma.make_mask( alpha < 1e-5 )
		self.Y = Y
		self.d = d
		self.beta = numpy.atleast_2d( numpy.ma.array( alpha, mask=mask ).compressed() )
		self.SV = numpy.atleast_2d( numpy.ma.array( numpy.tile(X,kappa), mask=mask).compressed() )
		self.gamma = numpy.atleast_2d( numpy.ma.array( Gamma, mask=mask ).compressed() )
		self.K = K
		
		duration = datetime.datetime.now() - start
		print "optimized in %ss" % (float(duration.microseconds)/1000000)
		print "%s SV found" % len(self.SV)
		print p
		
		
def run():
	mod = svm( array([[gauss(0,1)] for i in range(20) ] + [[gauss(8,1)] for i in range(20) ]).reshape([40,1]) )
	
	print "Total Loss: %s" % sum( (mod.Y.reshape( [len(mod.data),]) - mod.cdf( mod.data.reshape( [len(mod.data),]) ) ) ** 2)
	
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
	c.plot(numpy.sort(mod.data,0), numpy.sort(mod.Y,0), 'green' )
	c.plot(X, mod.cdf(X), 'r--' )
	#c.plot( mod.data, (mod.Y.reshape( [len(mod.data),]) - mod.cdf( mod.data.reshape( [len(mod.data),]) ) ) ** 2, '+' )
	c.set_title("Computed vs emprical CDF")
	
	d = fig.add_subplot(2,2,4)
	for i in range( len(mod.beta) ):
		d.plot( X, numpy.dot( mod._K( atleast_2d(X).T, mod.SV[i], mod.gamma[i] ), mod.beta[i].T ) )
		#d.plot( X, numpy.dot( mod._K( X, mod.SV[i], mod.gamma[i] ), mod.beta[i] ) )
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
