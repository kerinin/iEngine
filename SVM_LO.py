#! /usr/bin/env python

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
	def __init__(self,data=list(),C =1., Lambda = 1.0, gamma =.5):
		self.data = data
		self.Fl = None
		self.SV = None
		self.beta = None
		
		self.C = C
		self.Lambda = Lambda
		self.gamma = gamma
		
		self._compute()
	
	def _K(self,X,Y,gamma):
		# K(x_i,x_j) = 1/(\sqrt(2 \pi det( \Lambda ) ) ) exp( -.5(x_i - x_j) \Lambda^-1 (x_i - x_j)^T
		return ( 1 / ( 1+ exp( gamma * ( X - Y ) ) ) ).prod(2) 
		
	def Pr(self,x):
		# f(x) = \sum_{i=1}^N \beta_i \mathcal{K}(x_i, x)
		# \mathcal{K}(x,y) = \frac{ \gamma }{ 2 + e^{\gamma (x-y)} + e^{-\gamma (x-y)} }
		# NOTE: extend this to multiple dimensions...
		#return ( self.beta * ( self.gamma / ( 2 + exp( self.gamma * ( self.SV - x ) ) + exp( -self.gamma * ( self.SV - x ) ) ) ) ).sum()
		return ( self.beta * self._K( self.SV.reshape(len(self.SV),1,1), x.reshape(1,1,1), self.gamma ) ).sum()
	
	def __iadd__(self, points):
		# overloaded '+=', used for adding a vector list to the module's data
		# 
		# @param points		A LIST of observation vectors (not a single ovservation)
		self.data += points
	
	def _compute(self):
		start = datetime.datetime.now()
		#UNKNOWN: what are \alpha^*, \xi ???   check Vapnik regression
		# * used to designate negative values
		
		# Given
		# ( (x_1,F_\ell(x_1),\epsilon_1),...,(x_\ell),F_\ell(x_\ell),\epsilon_\ell) )
		# \theta(x) = indicator function; 1 if positive, 0 otherwise
		# F_\ell(x) = \frac{1}{\ell} \sum_{i=1}^{\ell} \theta(x - x_i)		NOTE: if d>1, \theta returns 0 if any dimension less than 0
		# \epsilon = \lambda \sigma_i = \lambda \sqrt{ \frac{1}{\ell} F_\ell(x_i)(1-F_\ell(x_i) ) }
		
		Kcount = 1
		C = self.C
		Lambda = self.Lambda
		gamma = self.gamma
		(N,d) = self.data.shape
		X = self.data

		Fl = ( (X.reshape(N,1,d) > transpose(X.reshape(N,1,d),[1,0,2])).prod(2).sum(1,dtype=float) / N ).reshape([N,1])
		e = Lambda * sqrt( (1/N) * ( Fl ) * (1-Fl) ).reshape([N,1])
		
		self.Fl = Fl
		
		# K(x,y) = \frac{1}{ 1 + e^{\gamma(x-y) } }
		# \mathcal{K}(x,y) = \frac{ \gamma }{ 2 + e^{\gamma (x - y) } + e^{-\gamma(x-y) } }

		# (Alternate - Multi-dimensional case)
		# K(x,y) = \prod{i=1}^d \frac{1}{ 1 + e^{\gamma(x^i-y^i) } }			!!NOTE: this is the L1 norm, not the L2
		# \mathcal{K}(x,y) = \prod_{i=1]^d \frac{ \gamma }{ 2 + e^{\gamma (x^i - y^i) } + e^{-\gamma(x^i-y^i) } }	
		# NOTE: this is highly suspect, based on "Multi-dimensional kernels can be chosen to be tensor products of one-dimensional kernels"
		# See Vapnik p.193 for discussion
		# He mentions "The kernel that defines the inner product in the n-dimensional basis is the product of one-dimensional kernels"
		# I'm not sure if this applies in this case, since we're using the L1 norm, and this may not constitude an inner product
		
		K = self._K( X.reshape(N,1,d), transpose(X.reshape(N,1,d), [1,0,2]), gamma )
		K1 = self._K( X.reshape(N,1,d), 1.0 , gamma ).reshape([1,N])
		K0 = self._K( X.reshape(N,1,d), 0.0 , gamma ).reshape([1,N])
		
		# Solve for
		# p(x) = \sum_{i=1}^N \alpha_i \mathcal(K)(x_i,x)		(N is the number of SV)
		
		# Minimize
		# \sum_{i=1}^\ell \alpha_i + C \sum_{i=1}^\ell \xi_i + C \sum_{i=1}^\ell \xi_i^*
		
		# (Alternate - kernel mixture)
		# \sum_{i=1}^\ell \sum{n=1}^K w_n \alpha_i^n + C \sum_{i=1}^\ell \xi_i + C \sum_{i=1}^\ell \xi_i^*
		# Note: w_n = n, if kernels are sequenced small to large width
		alpha = cvxmod.optvar( 'alpha',N,1)
		alpha.pos = True
		xipos = cvxmod.optvar( 'xi+',N,1)
		xipos.pos = True
		xineg = cvxmod.optvar( 'xi-',N,1)
		xineg.pos = True
		objective = cvxmod.minimize( cvxmod.sum(alpha) + C*cvxmod.sum(xipos) + C*cvxmod.sum(xineg) )
		
		# Subject To
		# y_i - \epsilon - \xi \le \sum_{j=1}^\ell \alpha_j K(x_i,x_j) \le y_i + \epsilon + \xi_i^*
		# \sum_{i=1}^\ell \alpha_i K(x_i,0) = 0
		# \sum_{i=1}^\ell \alpha_i k(x_i,1) = 1
		# \alpha_i, \xi_i, \xi_i^*  \ge 0
		
		# (Alternate - kernel mixture)
		# y_i - \epsilon - \xi \le \sum_{j=1}^\ell \sum_{n=1}^k \alpha_j^n K(x_i,x_j) \le y_i + \epsilon + \xi_i^*
		# \sum_{i=1}^\ell \sum_{n=1}^k \alpha_i^n k(x_i,1) = 1
		# \alpha_i, \xi_i, \xi_i^*  \ge 0			
		ineq1 = cvxopt.matrix( K ) * alpha <= cvxopt.matrix( Fl + e ) + xineg
		ineq2 = cvxopt.matrix( K ) * alpha >= cvxopt.matrix( Fl - e ) - xipos
		eq1 = cvxopt.matrix( K1, (1,N) ) * alpha == cvxopt.matrix(1.0)
		eq2 = cvxopt.matrix( K0, (1,N) ) * alpha == cvxopt.matrix(0.0)
		
		# Solve!
		p = cvxmod.problem( objective = objective, constr = [ineq1,ineq2,eq1,eq2] )
		
		start = datetime.datetime.now()
		p.solve()
		duration = datetime.datetime.now() - start
		print "optimized in %ss" % (float(duration.microseconds)/1000000)
		
		beta = ma.masked_less( alpha.value, 1e-7 )
		mask = ma.getmask( beta )
		data = ma.array(X,mask=mask)
		
		self.beta = beta.compressed()
		self.SV = data.compressed()
		print "%s SV's found" % len(self.SV)
		
def run():
	mod = svm( array([[gauss(0,1)] for i in range(100) ]).reshape([100,1]) )
	
	X = arange(-5.,5.,.05)
	Y_cmp = [ mod.Pr(x) for x in X ]
	
	#n, bins, patches = plt.hist(mod.data, 40, normed=1, facecolor='green', alpha=0.5)
	#bincenters = 0.5*(bins[1:]+bins[:-1])
	#plt.plot(bincenters, n, 'r', linewidth=1)
	
	#plt.plot(X,Y_cmp,'r--')
	
	#plt.plot( mod.SV, [ mod.Pr(x ) for x in  mod.SV ], 'o' )
	plt.plot(mod.data,mod.Fl, 'o' )
	plt.plot(X,Y_cmp, 'r--')
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
