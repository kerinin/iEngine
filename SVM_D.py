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
	def __init__(self,data=list(),C=1e-1, gamma =[ (2./3.)**i for i in range(-2,-1) ] ):
		self.data = data
		self.Fl = None
		self.SV = None
		self.betas = None
		
		self.C = C
		self.gamma = gamma
		
		self._compute()
	
	def _K(self,X,Y,gamma):
		diff = X - Y
		N = X.size
		M = Y.size
		
		# Sigmoid
		return [ ( 1 / ( 1 + exp( gi * diff ) ) ).reshape(N,M) for gi in gamma ]
		
		# RBF
		#return [ ( exp( -(diff**2) / gi ) ).reshape(N,M) for gi in gamma ]

	def cdf(self,x):
		ret = zeros(x.shape)
		
		# Inelegant I know, but for now...
		for i in range( len(self.gamma) ):
			gamma = self.gamma[i]
			beta = self.betas[i].compressed()
			data = numpy.ma.array(self.data, mask=numpy.ma.getmask(self.betas[i])).compressed()
			
			ret += numpy.dot( self._K( data.reshape([len(data),1]), x, [gamma,] )[0].T, beta )
		return ret
		
	def Pr(self,x):
		ret = zeros(x.shape)
		
		# Inelegant I know, but for now...
		#for i in range( len(self.gamma) ):
		for i in range( len(self.gamma) ):
			gamma = self.gamma[i]
			beta = self.betas[i].compressed()
			data = numpy.ma.array(self.data, mask=numpy.ma.getmask(self.betas[i])).compressed()
			diff = data.reshape([len(data),1]) - x
			
			ret += numpy.dot( beta.T, ( gamma / ( 2 + exp( gamma * diff ) + exp( -gamma * diff ) ) ) )
			
		return ret
		
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
		
		# Which means we don't need to calculate epsilon and we can eliminate the xi varaibles
		# In this case y_i = Xcmf_i and (I think) lambda is the same as described earlier, and
		# can be set to 1 for now
		
		# Gameplan: implement this minimization problem - if it works figure out what the matrix
		# definitions will be for the optimization problem and re-implement it in CVXOPT.  From
		# there you can start working on decomposition.

		C = self.C
		gamma = self.gamma
		Kcount = len( gamma )
		(N,d) = self.data.shape
		X = self.data
		
		# CMF of observations X
		Xcmf = ( (X.reshape(N,1,d) > transpose(X.reshape(N,1,d),[1,0,2])).prod(2).sum(1,dtype=float) / N ).reshape([N,1])
		
		K = self._K( Xcmf.reshape(N,1,d), transpose(Xcmf.reshape(N,1,d), [1,0,2]), gamma )
		
		pY = cvxmod.param("Y", value=cvxopt.matrix( Xcmf, ( 1, N ) ) )
		pY.pos = True
			
		alphas = list()
		Ks = list()
		expr1 = 0
		expr2 = 0
		eq = 0

		for i in range( Kcount ):
			alpha = cvxmod.optvar( 'alpha(%s)' % i, 1,N)
			alpha.pos = True
			pK = cvxmod.param('K(%s)' % i, value=cvxopt.matrix( K[i], ( N, N ) ) )
			pK.pos = True
			
			alphas.append( alpha )
			Ks.append(pK)
			expr1 += alpha * pK
			expr2 += gamma[i] * alpha
			eq += cvxmod.sum( alpha )
			
		#objective = cvxmod.minimize( cvxmod.sum( cvxmod.atoms.square( pY - expr1 ) ) + ( C * cvxmod.sum( expr2 ) ) )
		objective = cvxmod.minimize( cvxmod.sum( cvxmod.atoms.square( pY - expr1 ) ) )
		
		eq1 = eq == cvxopt.matrix( 1.0 )
		
		# Solve!
		p = cvxmod.problem( objective = objective, constr = [] )
		
		start = datetime.datetime.now()
		p.solve()
		p.classify()
		
		duration = datetime.datetime.now() - start
		print "optimized in %ss" % (float(duration.microseconds)/1000000)
		
		self.Fl = Xcmf
		self.betas = [ ma.masked_less( alpha.value, 1e-4).T for alpha in alphas ]
		
		print "SV's found: %s" % [ len( beta.compressed()) for beta in self.betas ]
			
		for i in range(len(self.betas)):
			beta = self.betas[i]
			if beta.count():
				print "SV @ gamma=%s: %s" % (self.gamma[i], str(ma.sort( ma.array(self.data, mask=ma.getmask(beta) ).compressed() ) ) )
		
def run():
	mod = svm( array([[gauss(0,1)] for i in range(20) ] + [[gauss(8,1)] for i in range(20) ]).reshape([40,1]) )
		
	print "Total Loss: %s" % sum( (mod.Fl.reshape( [len(mod.data),]) - mod.cdf( mod.data.reshape( [len(mod.data),]) ) ) ** 2)
	
	fig = plt.figure()
	
	start = -5.
	end = 12.
	X = arange(start,end,.25)
	
	a = fig.add_subplot(2,2,1)
	n, bins, patches = a.hist(mod.data, 20, normed=1, facecolor='green', alpha=0.5, label='empirical distribution')
	a.plot(X,mod.Pr(X), 'r--', label="computed distribution")
	a.set_title("Computed vs empirical PDF")
	
	c = fig.add_subplot(2,2,2)
	c.plot(numpy.sort(mod.data,0), numpy.sort(mod.Fl,0), 'green' )
	c.plot(X, mod.cdf(X), 'r--' )
	c.plot( mod.data, (mod.Fl.reshape( [len(mod.data),]) - mod.cdf( mod.data.reshape( [len(mod.data),]) ) ) ** 2, '+' )
	c.set_title("Computed vs emprical CDF")
	
	d = fig.add_subplot(2,2,4)
	for i in range(len(mod.betas) ):
		beta = mod.betas[i]
		
		for j in range(len(mod.data) ):
			if beta[j][0]:
				d.plot( X, beta[j][0] * mod._K(mod.data[j], X, [mod.gamma[i],])[0].reshape([len(X),1]) )
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
