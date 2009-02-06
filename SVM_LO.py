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
		
		# Given data
		# ( (x_1,F_\ell(x_1),\epsilon_1),...,(x_\ell),F_\ell(x_\ell),\epsilon_\ell) )
		
		# Solve for
		# p(x) = \sum_{i=1}^\ell \beta_i \mathcal(K)(x_i,x)
		
		# Minimize
		# \sum_{i=1}^\ell \alpha_i + \sum_{i=1}^\ell \alpha_i^* + C \sum_{i=1}^\ell \xi_i + \sum_{i=1}^\ell \xi_i^*
		
		# Subject To
		# y_i - \epsilon - \xi \le ( \sum_{j=1}^\ell (\alpha_j^* - \alpha_j) k(x_i,x_j) ) + b \le y_i + \epsilon + \xi_i^*
		# \alpha_i, \xi_i, \alpha_i^*, \xi_i^*  \ge 0
		
		# Where
		# \theta(x) = indicator function; 1 if positive, 0 otherwise
		# F_\ell(x) = \frac{1}{\ell} \sum_{i=1}^{\ell} \theta(x - x_i)
		# \epsilon = \lambda \sigma_i = \lambda \sqrt{ \frac{1}{\ell} F_\ell(x_i)(1-F_\ell(x_i) ) }
		# k(x,x') = \frac{1}{ 1 + e^{-\gamma(x-x') } }			!!NOTE: this is the L1 norm, not the L2
		# \mathcal{K}(x,x') = \frac{ \gamma }{ 2 + e^{\gamma (x - x') } + e^{-\gamma(x-x') } }
				
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
