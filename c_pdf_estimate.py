#! /usr/bin/env python

import sys, getopt, math, datetime, os
from math import sqrt

from numpy import *
from pylab import plot,bar,show,legend,title,xlabel,ylabel,axis

from cvxopt.base import *
from cvxopt.blas import dot 
from cvxopt.solvers import qp

from cvxopt import solvers
#solvers.options['show_progress'] = False

from santa_fe import getData

_Functions = ['run']
	
def sign(x):
	if isinstance(x, (int, long, float)):
		return int( x > 0 )
	else:
		return int( x.min() > 0)
		#for i in x:
		#	if i <= 0:
		#		return 0
		#return 1
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
		signmatrix = matrix( [ sign(i-self.x[k])*sign(j-self.y[k]) for k in range(len(self.x)) ] )
		return sum(signmatrix)/len(self.x)
	
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
					p_matrix[i,j] = self.beta[i]*(self.kernel.xx[j,i]*sign(self.x[p]-self.x[j])*
					self.kernel.int(p,i)-self.xy(self.x[p],self.y[p]))/self.l
			c_matrix[p,0] = sum(p_matrix)
		return c_matrix

class kernel:
	def __init__(self,x,y,gamma):
		# set variables
		self.gamma = gamma
		self.x = y
		self.y = y
		self.xx = matrix(0.0,(len(x),len(x)))
		self.xy = matrix(0.0,(len(x),len(y)))
		self.yy = matrix(0.0,(len(y),len(y)))
		
		N = len(x)
		# calculate matrix
		for i in range(len(x)):
			print 'x_ (%s, n) of %s calculated' % (i,N)
			for j in range(len(x)):
				if j>=i:
					val = self._calc(self.x[i],self.x[j])
					self.xx[i,j] = val
					self.xx[j,i] = val
			for j in range(len(y)):
				if j>=i:
					val = self._calc(self.x[i],self.y[j])
					self.xy[i,j] = val
					self.xy[j,i] = val
		f=open('xx.matrix','w')
		self.xy.tofile(f)
		f.close()
		f=open('xy.matrix','w')
		self.xy.tofile(f)
		f.close()
		print 'xy saved to file'
		for i in range(len(y)):
			print 'y_ (%s, n) of %s calculated' % (i,N)
			print i
			for j in range(len(y)):
				if j>=i:
					val = self._calc(self.y[i],y[j])
					self.yy[i,j] = val
					self.yy[j,i] = val
		f=open('yy.matrix','w')
		self.yy.tofile(f)
		f.close()
		print 'yy saved to file'

		# Normalize
		self.xx /= sum(self.xx)
		self.xy /= sum(self.xy)
		self.yy /= sum(self.yy)

	def int(self,i,j):
		# \int_{-\infty}^{y_i} K_\gamma{y_i,y_j}dy_i
		# When y_i is a vector of length 'n', the integral is a coordinate integral in the form
		# \int_{-\infty}^{y_p^1} ... \int_{-\infty}^{y_p^n} K_\gamma(y',y_i) dy_p^1 ... dy_p^n
		retval = 0
		for n in range(len(self.y)):
			if sign(self.y[i]-self.y[n]):
				retval += self.yy[n,j] 
		return retval

	def _calc(self,a,b):
	 	k = math.exp(-linalg.norm((a-b)/self.gamma))
		return k

def run():
	# Retrieve dataset
	data = getData('B1.dat')[:100]
	
	# Construct Variables
	gamma = 1.0
	N = len(data)-1
	sigma = 50/sqrt(N)
	K = kernel(data[:-1],data[1:],gamma)
	F = estimate(data[:-1],data[1:],K)
	
	# Objective Function
	print 'constructing objective function...'
	P = matrix(0.0,(N,N))
	for m in range(N):
		for n in range(N):
			if n >= m:
				P[m,n] = K.xx[n,m]*K.yy[n,m]
	q = matrix(0.0,(N,1))

	# Equality Constraint
	print 'constructing equality constraints...'
	A = matrix(0.0, (1,N))
	for n in range(N):
		A[n] = sum(matrix( [ K.xx[i,n] for i in range(N) ] ) ) / N
	#A = matrix(0.0, (N,N))
	#for m in range(N):
	#	for n in range(N):
	#		A[m,n] = K.xx[m,n] / N
	#b = matrix(1.0,(N,1))
	b = matrix(1.0)
	
	# Inequality Constraint
	print 'construction inequality constraints...'
	G = matrix(0.0, (N,N))
	for m in range(N):
		print "Inequality (%s,n) of %s calculated" % (m,N)
		for n in range(N):
			if n>= m:
				# CHECK THIS MATH - it's probably wrong, but you get the point
				K_int = K.int(n,m)
				def f(a,x):
					Kxx = K.xx[m::N]
					xn = resize(array(K.x[n]),(N,1))
					xx = array(K.x)
					return xx*K_int*(xn>xx)
				a=fromfunction(f,(1,N))
				
				G[n,m] = sum(a)/N - F.xy(data[n],data[n])
	h = matrix(sigma, (N,1))

	# Optimize
	print 'starting optimization...'
	print 'P.size = %s' % repr(P.size)
	print 'q.size = %s' % repr(q.size)
	print 'G.size = %s' % repr(G.size)
	print 'h.size = %s' % repr(h.size)
	print 'A.size = %s' % repr(A.size)
	print 'b.size = %s' % repr(b.size)
	optimized = qp(P, q,G= G, h=h, A=A, b=b)
	F.beta = optimized['x']
	f=open('beta.matrix','w')
	F.beta.tofile(f)
	f.close()
	print 'beta saved to file'

	# Display Results
	print 'optimized'
	print 'data points: %s' % N

	
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
