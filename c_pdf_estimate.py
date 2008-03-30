#! /usr/bin/env python

import sys, getopt, math, datetime
from math import sqrt

from numpy import *
from pylab import plot,bar,show,legend,title,xlabel,ylabel,axis

from cvxopt.base import matrix
from cvxopt.blas import dot 
from cvxopt.solvers import qp

from santa_fe import getData

_Functions = ['run']
	
def run():
	# Retrieve dataset
	data = getData('B1.dat')[:5]	
	
	# Construct Variables
	gamma = 1.0
	N = len(data)-1
	sigma = 50/sqrt(N)
	K = kernel(data[:-1],data[1:],gamma)
	
	# Objective Function
	P = matrix(0.0,(N,N))
	for m in range(N):
		for n in range(n):
			P[m][n] = K(data[n],data[m],gamma)*K(data[n+1],data[m+1],gamma)
	q = matrix(0,(1,N))

	# Equality Constraint
	A = matrix(0.0, (1,N))
	for n in range(N):
		A[n] = sum(matrix( K(data[i],data[n]) for i in range(N) )) / N
	b = matrix(1.0)
	
	# Inequality Constraint
	G = matrix(0.0, (n,n))
	for m in range(N):
		for n in range(N):
			tmp = matrix( [ (K.xx(i,m)*sign(data[n]-data[i])*K.int(n+1,m+1)) for i in range(N) ] )
			G[m][n] = sum(tmp)/N - F(data[n],data[n+1],gamma)
	h = matrix(sigma, (N,1))


	# Optimize
	#optimized = qp(P, q, G, h, A, b)
	print P
	print q
	print G
	print h
	print A
	print b

	# Display Results

def sign(x):
	if isinstance(x, (int, long, float)):
		return int( x > 0 )
	else:
		for i in x:
			if i <= 0:
				return 0
		return 1
class estimate:
	def init(self,x,y):
		# set variables
		if len(x) != len(y):
			raise StandardError, 'input/output values have different cardinality'
		self.x = matrix(x)
		self.y = matrix(y)

	def xy(i,j):
		signmatrix = matrix( [ sign(i-self.x[k])*sign(j-self.y[k]) for k in range(x) ] )
		sum(signmatrix)/len(x)
class kernel:
	def init(self,x,y,gamma):
		# set variables
		self.gamma = gamma
		self.x = x
		self.y = y
		self.xx = matrix(0.0,(len(x),len(x)))
		self.xy = matrix(0.0,(len(x),len(y)))
		self.yy = matrix(0.0,(len(y),len(y)))

		# calculate matrix
		for i in range(len(x)):
			for j in range(len(x)):
				self.xx[i][j] = self.calc(x[i],x[j])
			for j in range(len(y)):
				self.xy[i][j] = self.calc(x[i],y[j])
		for i in range(len(y)):
			for j in range(len(y)):
				self.yy[i][j] = self.calc(y[i],y[j])

		# Normalize
		self.xx /= sum(self.xx)
		self.xy /= sum(self.xy)
		self.yy /= sum(self.yy)

	def xx(self,i,j):
		return self.xx[i][j]
	def xy(self,i,j):
		return self.xy[i][j]
	def yy(self,i,j):
		return self.yy[i][j]
	def int(self,i,j):
		# \int_{-\infty}^{y_i} K_\gamma{y_i,y_j}dy_i
		# When y_i is a vector of length 'n', the integral is a coordinate integral in the form
		# \int_{-\infty}^{y_p^1} ... \int_{-\infty}^{y_p^n} K_\gamma(y',y_i) dy_p^1 ... dy_p^n
		a=array()
		for n in range(len(self.y)):
			if sign(self.y[i]-self.y[n]):
				a.append(self.yy(n,j))
		return sum( matrix(a) )

	def _calc(self,a,b):
		return math.exp(-abs((a-b)/self.gamma))

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
