#! /usr/bin/env python

import sys, getopt, math, datetime, os, cmath
from random import gauss

import numpy
import scipy
import scipy.special
import scipy.stats

from numpy import *
from pylab import *

_Functions = ['run']
	
class svm:
	def __init__(self,data=list(),C = .01, epsilon =.01, rho = 1e-2, L = 1):
		self.data = data
		self.W = None
		
		self.C = C
		self.epsilon = epsilon
		
		self._compute()
	
	def _K(self,X,Y):
		# K(x_i,x_j) = 1/(\sqrt(2 \pi det( \Lambda ) ) ) exp( -.5(x_i - x_j) \Lambda^-1 (x_i - x_j)^T
		#return ( 1. / ( sqrt( 2. * math.pi * self.L ) ) )* exp( -.5 * ( X - Y ) * self.L * ( X - Y ).T )  
		#return exp( ( -1.*(X-Y)**2 ) / ( 2*self.L*self.L ) )
		return ( 1. / ( sqrt( 2. * math.pi * self.L ) ) )* exp( -( X-Y )**2/self.L )
		
	def Pr(self,x):
		# x = [N,d]
		# \langle y(x) \rangle = \sum_{i=1}{N} w_i K(x,x_i)
		return ( self.W * self._K(x.T,self.data) ).sum(0)
	
	def __iadd__(self, points):
		# overloaded '+=', used for adding a vector list to the module's data
		# 
		# @param points		A LIST of observation vectors (not a single ovservation)
		self.data += points
	
	def _compute(self):
		start = datetime.datetime.now()
		#UNKNOWN: what are \alpha^*, \xi ???   check Vapnik regression
		# * used to designate negative values
		
		# Given data
		# ( (x_1,F_\ell(x_1),\epsilon_1),...,(x_\ell),F_\ell(x_\ell),\epsilon_\ell) )
		
		# Solve for
		# p(x) = \sum_{i=1}^N \alpha_i \mathcal(K)(x_i,x)		(N is the number of SV)
		
		# Minimize
		# \sum_{i=1}^\ell \alpha_i + C \sum_{i=1}^\ell \xi_i + \sum_{i=1}^\ell \xi_i^*
		
		# (Alternate - kernel mixture)
		# \sum_{i=1}^\ell \sum{n=1}^K w_n \alpha_i^n + C \sum_{i=1}^\ell \xi_i + \sum_{i=1}^\ell \xi_i^*
		# Note: w_n = n, if kernels are sequenced small to large width
		
		# Subject To
		# y_i - \epsilon - \xi \le \sum_{j=1}^\ell \alpha_j K(x_i,x_j) \le y_i + \epsilon + \xi_i^*
		# \sum_{i=1}^\ell \alpha_i K(x_i,0) = 0
		# \sum_{i=1}^\ell \alpha_i k(x_i,1) = 1
		# \alpha_i, \xi_i, \xi_i^*  \ge 0
		
		# (Alternate - kernel mixture)
		# y_i - \epsilon - \xi \le \sum_{j=1}^\ell \sum_{n=1}^k \alpha_j^n K(x_i,x_j) \le y_i + \epsilon + \xi_i^*
		# \sum_{i=1}^\ell \sum_{n=1}^k \alpha_i^n k(x_i,1) = 1
		# \alpha_i, \xi_i, \xi_i^*  \ge 0		
		
		# Where
		# \theta(x) = indicator function; 1 if positive, 0 otherwise
		# F_\ell(x) = \frac{1}{\ell} \sum_{i=1}^{\ell} \theta(x - x_i)		NOTE: if d>1, \theta returns 0 if any dimension less than 0
		# \epsilon = \lambda \sigma_i = \lambda \sqrt{ \frac{1}{\ell} F_\ell(x_i)(1-F_\ell(x_i) ) }
		
		# K(x,y) = \frac{1}{ 1 + e^{\gamma(x-y) } }
		# \mathcal{K}(x,y) = \frac{ \gamma }{ 2 + e^{\gamma (x - y) } + e^{-\gamma(x-y) } }
		
		# (Alternate - Multi-dimensional case)
		# K(x,y) = \prod{i=1}^d \frac{1}{ 1 + e^{\gamma(x^i-y^i) } }			!!NOTE: this is the L1 norm, not the L2
		# \mathcal{K}(x,y) = \prod_{i=1]^d \frac{ \gamma }{ 2 + e^{\gamma (x^i - y^i) } + e^{-\gamma(x^i-y^i) } }	
		# NOTE: this is highly suspect, based on "Multi-dimensional kernels can be chosen to be tensor products of one-dimensional kernels"
		# See Vapnik p.193 for discussion
		# He mentions "The kernel that defines the inner product in the n-dimensional basis is the product of one-dimensional kernels"
		# I'm not sure if this applies in this case, since we're using the L1 norm, and this may not constitude an inner product
				
		print "*** Optimization completed in %ss" % (datetime.datetime.now() - start).seconds
		print self.W
		print "%s Support Vectors of %s" % ( (self.W > 0).sum(), len(self.data) )
		
def run():
	mod = svm( array([[gauss(0,1)] for i in range(100) ]).reshape([100,1]) )
	
	X = frange(-5.,5.,.25)
	
	Y_cmp = mod.Pr(array(X).reshape([len(X),1]))
	
	Y_act = [ scipy.stats.norm.pdf(x) for x in X ]
	
	plot(X,Y_act,label="normal distribution")
	plot(X,Y_cmp,label="computed distribution")
	legend()
	show()
	
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
