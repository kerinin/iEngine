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
		self.SV = None
		
		self.C = C
		self.epsilon = epsilon
		self.rho = rho
		self.L = L
		
		self._compute()
	
	def _K(self,X,Y):
		# kernel from QP2
		return ( 1. /( self.L * sqrt( 2. *pi ) ) ) * exp( -( ( X - Y )**2 ) / (2. * ( self.L**2 ) ) )
		
	def Pr(self,x):
		# Pr from QP2
		return numpy.dot(self.W.T, self._K(self.SV,x) )
	
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
		
		X = self.data
		self.SV = self.data
		(N,d) = self.data.shape
		
		#NOTE: copy Xcmf from QP2
		t = ( (X.reshape(N,1,d) > transpose(X.reshape(N,1,d),[1,0,2])).prod(2).sum(1,dtype=float) / N ).reshape([N,1])
		
		# Set learning rate \rho and randomly set w_i
		self.W = ( numpy.random.rand( len(self.data), 1) * .9 ) + .1
		
		# Calculate covariance matrix K and let \sigma_i^2 = K_{ii}
		# \Lambda = variable (gamma)
		# \sigma = K_{ii}
		K = self._K(self.data, self.data.T)
		sigma2 = K.diagonal().reshape( len(self.data), 1)

		# Solve!
		self.optimize(K)
		
		# Cleanup from QP2
		W = ma.masked_less( self.W, 1e-7 )
		mask = ma.getmask( W )
		data = ma.array(X,mask=mask)
		
		self.W = W.compressed().reshape([ 1, len(W.compressed()) ])
		self.SV = data.compressed().reshape([len(W.compressed()),1])
		print "%s SV's found" % len(self.SV)
		
	def inner(self):
		# Inner Loop
		# yX = \langle y(x) \rangle = \sum_{i=1}^N w_i K(x, x_i )
		yX = self.Pr(self.data).reshape([len(self.data),1])
		
		#yX = array( [ self.Pr(x) for x in self.data ] ).reshape([len(self.data),1])
		
		# yXi = \langle y(s) \rangle_i = yX - \sigma_i^2 w_i
		yXi = yX - ( sigma2 * self.W )
	
		# Fi = C/2 exp( C/2 ( 2yXi - 2t_i + 2\epsilon + C \sigma_i^2 ) )
		#	* ( 1 - erf( ( yXi - t_i + \epsilon + C \sigma_i^2 ) / sqrt( 2 \sigma_i^2 ) )
		#	- C/2 exp( C/2 ( -2yXi + 2 t_i + 2 \epsilon + C \sigma_i^2 ) )
		#	* ( 1 - erf( ( - yXi + t_i + \epsilon + C \sigma_i^2 ) / sqrt( 2 \sigma_i^2 ) )
		Fi = (
			self.C/2. * exp( (self.C/2.) * ( ( 2. * yXi ) - ( 2. * t ) + ( 2. * self.epsilon ) + ( self.C * sigma2 ) ) ) 
			* ( 1. - scipy.special.erf( ( yXi - t + self.epsilon + ( self.C * sigma2 ) ) / sqrt( 2. * sigma2 ) ) )
			- self.C/2. * exp( (self.C/2.) * ( ( -2. * yXi ) + ( 2. * t ) + ( 2. * self.epsilon ) + ( self.C * sigma2 ) ) )
			* ( 1. - scipy.special.erf( ( -yXi + t + self.epsilon + ( self.C * sigma2 ) ) / sqrt(2. * sigma2 ) ) )
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
			+ .5 * exp( (self.C/2.) * ( ( -2. * yXi ) + ( 2. * t ) + ( 2. * self.epsilon ) + ( self.C * sigma2 ) ) )
			* ( 1. - scipy.special.erf( ( -yXi + t + self.epsilon + ( self.C * sigma2 ) ) / sqrt( 2. * sigma2 ) ) )
			)
	
		# w_i = w_i + \rho ( ( F_i / G_i ) - w_i )
		W_delta = self.rho * ( Fi / Gi - self.W )
	
		self.W += W_delta
		return ( t, yXi, Gi, W_delta )
	
	def optimize(self,K):
		# Outer Loop
		while True:
			
			for i in range(2):
				self.inner()
			
			(t,yXi,Gi,W_delta) = self.inner()
			
			# IG_i = 1/2 erf( ( t_i - \leftangle y(x_i) \rightangle_i + \epsilon ) / sqrt( 2 \sigma_i^2 ) ) - 1/2 erf( ( t_i - \leftangle y(x_i) \rightangle_i - \epsilon ) )
			IG = .5 * scipy.special.erf( ( t - yXi + self.epsilon ) / sqrt( 2 * sigma2 ) ) - .5 * scipy.special.erf( ( t - yXi - self.epsilon ) / sqrt( 2 * sigma2 ) )
			
			# Z = C^2 - w_i^2 - \frac{ w_i \leftangle y(x_i) \rightangle_i  + \sigma_i^2 C^2 + IG_i }{ \sigma_i^2 G( \leftangle y(x_i) \rightangle_i, \sigma_i^2 ) }
			Z = ( self.C ** 2 ) - ( self.W ** 2 ) - ( ( ( self.W * yXi ) + ( sigma2 * ( self.C ** 2 ) ) + IG ) / ( sigma2 * Gi ) )
			
			# \Sigma_i = - \sigma_i^2 - ( Z )^{-1}
			Sigma_i = -sigma2 - ( 1 / Z )
			
			# \Sigma = diag( \Sigma_1,...,\Sigma_N )
			Sigma = diag( Sigma_i.reshape([len(self.data),] ) )
			
			# sigma_i^2 = \frac{ 1 } { [ ( \Sigma + K ) ^{-1} ]_{ii} } - \Sigma_i
			sigma2 = ( 1 / ( 1 / ( Sigma + K ) ).diagonal().reshape([len(self.data),1]) ) - Sigma_i
			
			if absolute(W_delta.max()) < .0005:
				break
			else:
				print "Cumulative adjustment of Coefficients: %s" % absolute(W_delta).sum()

def run():
	mod = svm( array([[gauss(0,1)] for i in range(100) ]).reshape([100,1]) )
	
	X = frange(-5.,5.,.25)
	
	Y_cmp = mod.Pr(array(X).reshape([len(X),1]))
	
	Y_act = [ scipy.stats.norm.pdf(x) for x in X ]
	
	plot(X,Y_act,label="normal distribution")
	plot(X,Y_cmp,label="computed distribution")
	legend()
	#show()
	
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
