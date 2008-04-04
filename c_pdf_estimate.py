#! /usr/bin/env python

import sys, getopt, math, datetime, os
from math import sqrt, sin

from numpy import *
from pylab import plot,bar,show,legend,title,xlabel,ylabel,axis

from cvxopt.base import *
from cvxopt.blas import dot 
from cvxopt.solvers import qp

from cvxopt import solvers
solvers.options['show_progress'] = False

from santa_fe import getData

_Functions = ['run']
	
def sign(x,y):
	if isinstance(x, (int, long, float)):
		return int( x > 0 )
	else:
		return int( sum(x>y) == len(x) )
		
class estimate:
	def __init__(self,x,y,kernel):
		# set variables
		if len(x) != len(y):
			raise StandardError, 'input/output values have different cardinality'
		self.l = len(x)
		self.x = x
		self.y = y
		self.kernel = kernel
		self.beta = None

	def xy(self,i,j):
	################################################################################
	#
	#	F_\ell(y,x) = frac{1}{\ell} \sum_{i=1}^{\ell} \theta(y-y_i) \theta(x-x_i)
	#
	# where y=i, x=j, l=self.l
	# and i,j are both vectors of x and y (not indices of training data)
	#
	################################################################################
	
		signmatrix = array( [ sign(i,self.x[k])*sign(j,self.y[k]) for k in range(self.l) ] )
		return sum(signmatrix)/self.l
	
	def r(self,x):
		ret = zeros(self.kernel.n)
		for i in range(self.kernel.l):
			ret += self.kernel.y[i]*self.beta[i]*self.kernel._calc(x,self.kernel.x[i])
		return ret
		
	def equality_check(self):
		c_matrix = matrix(0.0,(self.l,self.l))
		for i in range(self.l):
			for j in range(self.l):
				c_matrix[i,j] = self.beta[j]*self.kernel.xx[i,j]/self.l
		return sum(c_matrix)

	def inequality_check(self):
		c_matrix = matrix(0.0,(self.l,1))
		for p in range(self.l):
			p_matrix = matrix(0.0,(self.l,self.l))
			for i in range(self.l):
				for j in range(self.l):
					p_matrix[i,j] = self.beta[i]*(self.kernel.xx[j,i]*sign(self.x[p],self.x[j])*
					self.kernel.int(p,i)-self.xy(self.x[p],self.y[p]))/self.l
			c_matrix[p,0] = sum(p_matrix)
		return c_matrix

class kernel:
	def __init__(self,data,gamma,sigma_q):
		# set variables
		self.l = len(data)-1
		try:
			self.n = len(data[0])
		except TypeError:
			self.n = 1
		self.x = data[:-1]
		self.y = data[1:]
		self.xx = matrix(0.0,(self.l,self.l))
		self.yy = matrix(0.0,(self.l,self.l))
		self.intg = matrix(0.0,(self.l,self.l))
		self.gamma = gamma
		self.sigma = .5

		
		# calculate xx matrix
		for i in range(self.l):
			for j in range(i,self.l):
				val = self._calc(self.x[i],self.x[j])
				self.xx[i,j] = val
				self.xx[j,i] = val
		# normalize
		self.xx /= (sum(self.xx)/self.l)
		f=open('xx.matrix','w')
		self.xx.tofile(f)
		f.close()
		print 'xx saved to file'
		
		# calculate yy matrix
		for i in range(self.l):
			for j in range(i,self.l):
				val = self._calc(self.y[i],self.y[j])
				self.yy[i,j] = val
				self.yy[j,i] = val
		# normalize
		self.yy /= (sum(self.yy)/self.l)
		f=open('yy.matrix','w')
		self.yy.tofile(f)
		f.close()
		print 'yy saved to file'

		# calculate integration matrix
		print 'computing integrals...'
		for i in range(self.l):
			for j in range(i,self.l):
				val = self.int(i,j)
				self.intg[i,j] = val
				self.intg[j,i] = val
		f=open('intg.matrix','w')
		self.intg.tofile(f)
		f.close()
		print 'intg saved to file'

	def int(self,i,j):
		# \int_{-\infty}^{y_i} K_\gamma{y_i,y_j}dy_i
		# When y_i is a vector of length 'n', the integral is a coordinate integral in the form
		# \int_{-\infty}^{y_p^1} ... \int_{-\infty}^{y_p^n} K_\gamma(y',y_i) dy_p^1 ... dy_p^n
		# note that self.y is a vector array, while self.yy is a matrix of K values
		# 
		# After going over the math, the integral of the function should be calculated as follows
		# take the sum of K for all values of y which have at least one dimension less than y_p
		# times the inverse of lxn where l is the total number of y and n is the dimensionality of y
		
		# select the row (*,j) of self.yy 
		yi = self.yy[self.l*j:self.l*(j+1)]
		for n in range(self.l):
			# scale K according to how many dimensions are less than y_p 
			# ( note that this also zeroes out y which are larger than y_p)
			yi[n,0] = yi[n,0]*(sum(self.y[n]<self.y[i]))
			
		# return the sum of the remaining values of K divided by lxn where l is the number of y and n is the dimensionality
		return sum(yi)/(self.l*self.n)

	def _calc(self,a,b):
	 	return math.exp(-linalg.norm((a-b)/self.gamma))

def run():
	# Retrieve dataset
	#data = getData('B1.dat')[:80]
	data = array([sin(i/4.) for i in range(50)])
	
	# Construct Variables
	K = kernel(data,gamma=.1,sigma_q=.005)
	F = estimate(data[:-1],data[1:],K)
	
	# Objective Function
	#FIXME: check the math for all the remaining stuff
	print 'constructing objective function...'
	P = mul(K.xx,K.yy)
	q = matrix(0.0,(K.l,1))
	
	# Equality Constraint
	print 'constructing equality constraints...'
	A = matrix( [ sum( K.xx[ n*K.l:( n+1 )*K.l ] for n in range( K.l ) ) ], ( 1,K.l ) ) / K.l
	b = matrix(1.0)
	
	# Inequality Constraint
	print 'construction inequality constraints...'
	G = matrix(0.0, (K.l,K.l))
	for m in range(K.l):		
		print "Inequality (%s,n) of %s calculated" % (m,K.l)
		for n in range(m,K.l):
			k = K.xx[m::K.l]
			if K.n > 1:
				t = array( [min(K.x[n] - K.x[i]) > 0 for i in range(K.l)] )
			else:
				t = array( [K.x[n] - K.x[i] > 0 for i in range(K.l)])
			i = K.intg[m::K.l]
				
			G[n,m] = sum(k*t*i)/K.l - F.xy(K.x[n],K.y[n])
			G[m,n] = sum(k*t*i)/K.l - F.xy(K.x[n],K.y[n])
				
	h = matrix(K.sigma, (K.l,1))

	# Optimize
	print 'starting optimization...'
	optimized = qp(P, q,G=G, h=h, A=A, b=b)
	F.beta = optimized['x']
	print F.beta
	f=open('beta.matrix','w')
	F.beta.tofile(f)
	f.close()
	print 'beta saved to file'

	# test on training data
	x_1 = list()
	y_1 = list()
	
	for i in range(K.l):
		est = F.r(K.x[i])
		x_1.append( est)
		y_1.append( K.y[i])
		
	plot(x_1,label="x'")
	plot(y_1,label="y")
	legend()
	show()
	
def help():
	print __doc__
	return 0
	
def process(arg='run'):
	if arg in _Functions:
		globals()[arg]()
	class Usage(Exception):    def __init__(self, msg):        self.msg = msgdef main(argv=None):	if argv is None:		argv = sys.argv	try:		try:			opts, args = getopt.getopt(sys.argv[1:], "hl:d:", ["help","list=","database="])		except getopt.error, msg:			raise Usage(msg)
				# process options		for o, a in opts:			if o in ("-h", "--help"):
				for f in _Functions:
					if f in args:
						apply(f,(opts,args))
						return 0				help()
				# process arguments		for arg in args:			process(arg) # process() is defined elsewhere
	except Usage, err:		print >>sys.stderr, err.msg		print >>sys.stderr, "for help use --help"		return 2if __name__ == "__main__":	sys.exit(main())
