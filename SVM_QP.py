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
	def __init__(self,data=list(),C =1., Lambda = 1., gamma =.25):
		self.data = data
		self.Fl = None
		self.SV = None
		self.beta = None
		
		self.C = C
		self.Lambda = Lambda
		self.gamma = gamma
		
		self._compute()
	
	def _K(self,X,Y,gamma):
		# Gaussian kernel w/ width gamma
		# NOTE: this is only applicable to 1d data points
		return ( 1/( gamma * sqrt( 2*pi ) ) ) * exp( -( ( X - Y )**2 ) / (2 * ( gamma**2 ) ) )
		
	def Pr(self,x):
		return numpy.dot(self.beta,self.SV)[0][0]
	
	def __iadd__(self, points):
		# overloaded '+=', used for adding a vector list to the module's data
		# 
		# @param points		A LIST of observation vectors (not a single ovservation)
		self.data += points
	
	def _compute(self):
		start = datetime.datetime.now()

		Kcount = 1.
		C = self.C
		Lambda = self.Lambda
		gamma = self.gamma
		(N,d) = self.data.shape
		X = self.data

		Xcmf = ( (X.reshape(N,1,d) > transpose(X.reshape(N,1,d),[1,0,2])).prod(2).sum(1,dtype=float) / N ).reshape([N,1])
		sigma = median(X) / sqrt(N)
		
		K = self._K( X.reshape(N,1,d), transpose(X.reshape(N,1,d), [1,0,2]), gamma ).reshape([N,N])
		#NOTE: this integral depends on K being the gaussian kernel
		Kint =  ( (1.0/gamma)*scipy.special.ndtr( (X-X.T)/gamma ) )
		
		alpha = cvxmod.optvar( 'alpha',N,1)
		alpha.pos = True
		pK = cvxmod.param( 'K',N,N )
		pK.psd = True
		pK.value = cvxopt.matrix(K,(N,N) )
		pKint = cvxmod.param( 'Kint',N,N )
		pKint.value = cvxopt.matrix(Kint,(N,N))
		pKint.pos = True
		pXcmf = cvxmod.param( 'Xcmf',N,1)
		pXcmf.value = cvxopt.matrix(Xcmf, (N,1))
		pXcmf.pos = True
		
		objective = cvxmod.minimize( cvxmod.atoms.quadform(alpha, pK) )
		eq1 = pXcmf - ( pKint * alpha ) <= sigma
		eq2 = cvxmod.sum( alpha ) == 1.0
		
		# Solve!
		p = cvxmod.problem( objective = objective, constr = [eq1, eq2] )
		
		start = datetime.datetime.now()
		p.solve()
		duration = datetime.datetime.now() - start
		print "optimized in %ss" % (float(duration.microseconds)/1000000)
		
		beta = ma.masked_less( alpha.value, 1e-7 )
		mask = ma.getmask( beta )
		data = ma.array(X,mask=mask)
		
		self.Fl = Xcmf
		self.beta = beta.compressed().reshape([ 1, len(beta.compressed()) ])
		self.SV = data.compressed().reshape([len(beta.compressed()),1])
		print "%s SV's found" % len(self.SV)
		print self.SV
		
def run():
	mod = svm( array([[gauss(0,1)] for i in range(10) ]).reshape([10,1]) )
	
	X = arange(-5.,5.,.05)
	Y_cmp = [ mod.Pr(x) for x in X ]
	
	#n, bins, patches = plt.hist(mod.data, 40, normed=1, facecolor='green', alpha=0.5)
	#bincenters = 0.5*(bins[1:]+bins[:-1])
	#plt.plot(bincenters, n, 'r', linewidth=1)
	
	#plt.plot( mod.SV, [ mod.Pr(x ) for x in  mod.SV ], 'o' )
	#print Y_cmp
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
