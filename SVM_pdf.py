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
	def __init__(self,data=list(),C = .01, epsilon = .5, rho = 1e-4, L = .05):
		self.data = data
		self.W = None
		
		self.C = C
		self.epsilon = epsilon
		self.rho = rho
		self.L = L
		
		self._compute()
	
	def _K(self,X,Y):
		# K(x_i,x_j) = 1/(\sqrt(2 \pi det( \Lambda ) ) ) exp( -.5(x_i - x_j) \Lambda^-1 (x_i - x_j)^T
		return 1. / ( sqrt( 2. * self.L ) )* exp( -.5 * ( X - Y ) * ( X - Y ).T / self.L ) 
		
	def Pr(self,x):
		# \langle y(x) \rangle = \sum_{i=1}{N} w_i K(x,x_i)
		return ( self.W * self._K(x,self.data) ).sum()
	
	def __iadd__(self, points):
		# overloaded '+=', used for adding a vector list to the module's data
		# 
		# @param points		A LIST of observation vectors (not a single ovservation)
		self.data += points
	
	def _compute(self):
		# Set the training pairs
		# (x_1,F_N(x_1)),...,(x_N,F_N(x_N))
		# F_N = 1/N \sum_{k=1}^N I(x-x_k)
		# I = Indicator function (0 if negative, 1 otherwise)
		t = ( ( self.data.T > self.data ).sum(1,dtype=float) / len(self.data) ).reshape( len(self.data), 1)	# Since F is an estimate of the target value t, i'm simply renaming it
		
		# Set learning rate \rho and randomly set w_i
		W = numpy.random.rand( len(self.data), 1) * .9 + .1
		
		# Calculate covariance matrix K and let \sigma_i^2 = K_{ii}
		# \Lambda = variable (gamma)
		# \sigma = K_{ii}
		K = self._K(self.data, self.data.T)
		sigma2 = K.diagonal().reshape( len(self.data), 1)
		
		# Inner Loop
		def inner():

			# Inner Loop
			# yX = \langle y(x) \rangle = \sum_{i=1}^N w_i K(x, x_i )
			yX = K[0].reshape( [ len(self.data), 1] ) * W
			
			# yXi = \langle y(s) \rangle_i = yX - \sigma_i^2 w_i
			yXi = yX - sigma2 * W

			# Fi = C/2 exp( C/2 ( 2yXi - 2t_i + 2\epsilon + C \sigma_i^2 ) )
			#	* ( 1 - erf( ( yXi - t_i + \epsilon + C \sigma_i^2 ) / sqrt( 2 \sigma_i^2 ) )
			#	- C/2 exp( C/2 ( -2yXi + 2 t_i + 2 \epsilon + C \sigma_i^2 ) )
			#	* ( 1 - erf( -C/2 ( 2yXi - 2t_i + 2\epsilon + C \sigma_i^2 ) )
			Fi = (
				self.C/2. * exp( (self.C/2.) * ( ( 2. * yXi ) - ( 2. * t ) + ( 2. * self.epsilon ) + ( self.C * sigma2 ) ) ) 
				* ( 1. - scipy.special.erf( ( yXi - t + self.epsilon + ( self.C * sigma2 ) ) / sqrt( 2. * sigma2 ) ) )
				- self.C/2. * exp( (self.C/2.) * ( ( -2. * yXi ) + ( 2. * t ) + ( 2. * self.epsilon ) + sigma2 ) )
				* ( 1. - scipy.special.erf( ( -self.C / 2. ) * ( ( 2. * yXi ) - ( 2. * t ) + ( 2. * self.epsilon ) + ( self.C * sigma2 ) ) ) )
				)
				
			# Gi = 1/2 erf( ( t_i - yXi + \epsilon ) / sqrt( s \sigma_i^2 ) )
			#	- 1/2 erf( ( t_i - yXi - \epsilon ) / sqrt( s \sigma_i^2 ) )
			#	+ 1/2 exp( C/2 ( 2 yXi - 2 t_i + 2 \epsilon + C \sigma_i^2 ) )
			#	* ( 1 - erf( ( yXi - t_i + \epsilon + \C \sigma_i^2 ) / sqrt( 2 \sigma_i^2 ) )
			#	+ 1/2 exp( C/2 ( -2 yXi - 2 t_i + 2 \epsilon + C \sigma_i^2 ) )
			#	* ( 1 - erf( ( yXi - t_i + \epsilon + \C \sigma_i^2 ) / sqrt( 2 \sigma_i^2 ) )
			Gi = (
				.5 * scipy.special.erf( ( t - yXi + self.epsilon ) / sqrt( 2. * sigma2 ) )
				- .5 * scipy.special.erf( ( t - yXi - self.epsilon ) / sqrt( 2. * sigma2 ) )
				+ .5 * exp( (self.C/2.) * ( ( 2. * yXi ) - ( 2. * t ) + ( 2. * self.epsilon ) + ( self.C * sigma2 ) ) )
				* ( 1. - scipy.special.erf( ( yXi - t + self.epsilon + ( self.C * sigma2 ) ) / sqrt( 2. * sigma2 ) ) )
				+ .5 * exp( (self.C/2.) * ( ( -2. * yXi ) - ( 2. * t ) + ( 2. * self.epsilon ) + ( self.C * sigma2 ) ) )
				* ( 1. - scipy.special.erf( ( yXi - t + self.epsilon + ( self.C * sigma2 ) ) / sqrt( 2. * sigma2 ) ) )
				)

			# erf(x) = 2/sqrt(\pi) \sum_0^x e^{-t^2} dt (see scipy.special.erf)
			# w_i = w_i + \rho ( ( F_i / G_i ) - w_i )
			return ( t, yXi, Gi, nan_to_num( self.rho * ( Fi / Gi - W ) ) )
				
		start = datetime.datetime.now()
		
		# Outer Loop
		while True:
			
			for i in range(10):
				W += inner()[3]
				W *= W > 0
			
			(t,yXi,Gi,W_delta) = inner()
			
			# IG_i = 1/2 erf( ( t_i - \leftangle y(x_i) \rightangle_i + \epsilon ) / sqrt( 2 \sigma_i^2 ) ) - 1/2 erf( ( t_i - \leftangle y(x_i) \rightangle_i - \epsilon ) )
			IG = .5 * scipy.special.erf( ( t - yXi + self.epsilon ) / sqrt( 2 * sigma2 ) ) - .5 * scipy.special.erf( ( t - yXi - self.epsilon ) / sqrt( 2 * sigma2 ) )
			
			# Z = C^2 - w_i^2 - \frac{ w_i \leftangle y(x_i) \rightangle_i  + \sigma_i^2 C^2 + IG_i }{ \sigma_i^2 G( \leftangle y(x_i) \rightangle_i, \sigma_i^2 ) }
			Z = ( self.C * 2 ) - ( W ** 2 ) - ( ( ( W * yXi ) + ( sigma2 * ( self.C ** 2 ) ) + IG ) / ( sigma2 * Gi ) )
			
			# \Sigma_i = - \sigma_i^2 - ( Z )^{-1}
			Sigma_i = -sigma2 - ( 1 / Z )
			
			# \Sigma = diag( \Sigma_1,...,\Sigma_N )
			Sigma = diag( Sigma_i.reshape([len(self.data),] ) )
			
			# sigma_i^2 = \frac{ 1 } { [ ( \Sigma + K ) ^{-1} ]_{ii} } - \Sigma_i
			sigma2 = ( 1 / ( 1 / ( Sigma + K ) ).diagonal().reshape([len(self.data),1]) ) - Sigma_i
			
			if W_delta. max() < .0005:
				self.W = W + W_delta
				break
			else:
				print "Cumulative adjustment of Coefficients: %s" % absolute(W_delta).sum()
				W += W_delta
				
		#print "*** Optimization completed in %ss" % (datetime.datetime.now() - start).seconds
		print "%s Support Vectors of %s" % ( (W > 0).sum(), len(self.data) )
		self.W = W
		
def run():
	mod = svm( array([[gauss(0,1)] for i in range(100) ]).reshape([100,1]) )

	X = frange(-5.,5.,.1)
	Y_cmp = [ mod.Pr(x) for x in X ]
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
