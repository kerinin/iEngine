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

from svm import *
from numpy import *

import matplotlib.pyplot as plt

_Functions = ['run']
		
class svm:
	def __init__(self,X=list(),nu=.5, gamma =0.5 ):
		self.X = X
		self.Y = None
		self.SV = None
		self.beta = None
		self.pdf = None
		
		self.nu = nu
		self.gamma = gamma
		
		self._compute()

	def cdf(self,x):
		return self.mod.predict_values(x)
		
	def Pr(self,x):
		return x
		
	def __iadd__(self, points):
		# overloaded '+=', used for adding a vector list to the module's data
		# 
		# @param points		A LIST of observation vectors (not a single ovservation)
		self.data += points
	
	def _compute(self,path="output.svm"):
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
		
		start = datetime.datetime.now()
		
		X = self.X
		(N,d) = X.shape
		Y = ( (X.reshape(N,1,d) > transpose(X.reshape(N,1,d),[1,0,2])).prod(2).sum(1,dtype=float) / N )
		self.Y = Y
		
		print X.shape
		print Y.shape
		
		self.param = svm_parameter(svm_type=NU_SVR, kernel_type = SIGMOID, nu = self.nu, gamma = self.gamma, cache_size = 500)
		self.mod = svm_model(svm_problem( list(Y), list(X) ),self.param)
		self.mod.save(path)
		
		parse_file = file(path,'r')
		lines = parse_file.readlines()
		self.rho = float( lines[5].split(' ')[1] )
		
		SV = list()
		betas = list()
		for line in lines[8:]:
			text = line.split(' ')
			beta = float( text[0] )
			
			data = list()
			try:
				for value in text[1:]:
					v  = value.split(':')
					data.append( float(v[1]) )
			except IndexError:
				pass
			
			SV.append( data )
			betas.append( beta)
			
		self.SV = array(SV)
		self.beta = array(betas)
		#self.pdf = self.mod.get_svr_pdf()

		duration = datetime.datetime.now() - start
		print "optimized in %ss" % (float(duration.microseconds)/1000000)
		print "SV's found: %s" % len(SV)
					
def run():
	mod = svm( array([[gauss(0,1)] for i in range(20) ] + [[gauss(8,1)] for i in range(20) ]).reshape([40,1]) )
	
	print "Total Loss: %s" % sum( (mod.Y.reshape( [len(mod.X),]) - mod.cdf( mod.X.reshape( [len(mod.X),]) ) ) ** 2)
	
	fig = plt.figure()
	
	start = -5.
	end = 12.
	X = arange(start,end,.25)
	
	#a = fig.add_subplot(2,2,1)
	#n, bins, patches = a.hist(mod.data, 20, normed=1, facecolor='green', alpha=0.5, label='empirical distribution')
	#a.plot(X,mod.Pr(X), 'r--', label="computed distribution")
	#a.set_title("Computed vs empirical PDF")
	
	c = fig.add_subplot(2,2,2)
	c.plot(numpy.sort(mod.X,0), numpy.sort(mod.Y,0), 'green' )
	c.plot(X, [ mod.cdf((x,)) for x in X ], 'r--' )
	c.plot( mod.X, (mod.Y.reshape( [len(mod.X),]) - mod.cdf( mod.X.reshape( [len(mod.X),]) ) ) ** 2, '+' )
	c.set_title("Computed vs emprical CDF")
	
	#d = fig.add_subplot(2,2,4)
	#for i in range(len(mod.beta) ):
	#	beta = mod.beta[i]
		
	#	for j in range(len(mod.X) ):
	#		if beta[j][0]:
	#			d.plot( X, beta[j][0] * mod._K(mod.X[j], X, [mod.gamma[i],])[0].reshape([len(X),1]) )
	#d.set_title("SV Contributions")
	
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
