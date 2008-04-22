#! /usr/bin/env python

from system_2_base import input_base, function_base

import sys, getopt, math, datetime, os
from math import sqrt, sin

from numpy import *
from pylab import plot,bar,show,legend,title,xlabel,ylabel,axis

from cvxopt.base import *
from cvxopt.blas import dot 
from cvxopt.solvers import qp

from cvxopt import solvers
solvers.options['show_progress'] = False

def timedelta_float(td):
	return (td.days>1 and td.days or 0)*86400+td.seconds+float(td.microseconds)/1000000
	
def sign(x,y=0):
	if isinstance(x, (int, long, float)):
		return int( x > 0 )
	else:
		return int( array( x > y ).all() )
		
class kernel:
	l = 0							# the number of data points cached so far
	n = 1						# the dimensionality of the data (assumed to be one in all cases here)
	
	gamma = None				# smoothing variable
	sigma_q = None				# quantile to consider for the residual principal 
	sigma = 1.0					# calculated sigma value from last observation and given quantile
	
	def __init__(self,gamma=.5,sigma_q=.5):
		self.gamma = gamma
		self.sigma_q = sigma_q
		
	def load(self,observations):
		# set variables
		self.l = len(observations)
		self.x = list()
		self.y = list()
		
		for observation in observations:
			self.x.append(observation.t)
			self.y.append(observation.val)
			
		self.xx = matrix(0.0,(self.l,self.l))
		self.yy = matrix(0.0,(self.l,self.l))
		self.intg = matrix(0.0,(self.l,self.l))
		
		# calculate x distances
		for i in range(self.l):
			for j in range(i,self.l):
				val = self._calc(self.x[i],self.x[j])
				self.xx[i,j] = val
				self.xx[j,i] = val
		# normalize
		self.xx /= (sum(self.xx)/self.l)
		
		# calculate y distances
		for i in range(self.l):
			for j in range(i,self.l):
				val = self._calc(self.y[i],self.y[j])
				self.yy[i,j] = val
				self.yy[j,i] = val
		# normalize
		self.yy /= (sum(self.yy)/self.l)
		
		# calculate integrals
		for i in range(self.l):
			for j in range(i,self.l):
				val = self.int(i,j)
				self.intg[i,j] = val
				self.intg[j,i] = val
				
		return self
		
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
		
	def xy(self,i,j):
	#F_\ell(y,x) = frac{1}{\ell} \sum_{i=1}^{\ell} \theta(y-y_i) \theta(x-x_i)
	#
	# where y=i, x=j, l=self.l
	# and i,j are both vectors of x and y (not indices of training data)
	
		signmatrix = array( [ sign(i,self.x[k])*sign(j,self.y[k]) for k in range(self.l) ] )
		return sum(signmatrix)/self.l

	def _calc(self,a,b):
		try:
			return math.exp(-linalg.norm((a-b)/self.gamma))
		except TypeError:
			return math.exp(-linalg.norm(timedelta_float(a-b)/self.gamma))

		
class function_svm(function_base):
# implements functional estimation using the SVM architecture

	# kill = None			# the kill time of the function
	
	kernel = None		# kernel for this function
	SV = list()			# list of SV functions
	beta = list()			# list of SV multipliers
	_function_d = None	# list of computed distances to different functions
		
	def __init__(self,data=None,kernel=None,*args,**kargs):
		self.kernel = kernel
		function_base.__init__(self,data,*args,**kargs)
			
	def __sub__(self,a):
		raise StandardError, 'This function not implemented'
		
	def optimize(self,data):
		K = self.kernel.load(data)
		
		# construct objective functions
		P = mul(K.xx,K.yy)
		q = matrix(0.0,(K.l,1))	
		
		# construct equality constraints
		A = matrix( [ sum( K.xx[ n::K.l ] for n in range( K.l ) ) ], ( 1,K.l ) ) / K.l
		b = matrix(1.0)
	
		# construct inequality constraints
		G = matrix(0.0, (K.l,K.l))
		for m in range(K.l):		
			k = K.xx[m::K.l]
			
			for n in range(m,K.l):
				if K.n > 1:
					t =array( [min(K.x[n] - K.x[i]) > datetime.timedelta() for i in range(K.l)] )
				else:
					t = array( [K.x[n] - K.x[i] > datetime.timedelta() for i in range(K.l)])
				i = K.intg[m,n]
				
				G[n,m] = sum(k*t*i)/K.l - K.xy(K.x[n],K.y[n])
				G[m,n] = sum(k*t*i)/K.l - K.xy(K.x[n],K.y[n])
		h = matrix(K.sigma, (K.l,1))
		
		# optimize and set variables
		optimized = qp(P, q,G=G,h=h,A=A, b=b)
		for i in range(len(optimized['x'])):
			if optimized['x'][i]:
				self.SV.append( K.x[i] )
				self.beta.append( optimized['x'][i] )
		
		
	def reg(self,x):
		ret = zeros(self.kernel.n)
		for i in range(self.kernel.l):
			ret += self.kernel.y[i]*self.beta[i]*self.kernel._calc(x,self.kernel.x[i])
		return ret
		
	def den(self,t):
		
		raise StandardError, 'This function not implemented'
		
	def equality_check(self):
		c_matrix = matrix(0.0,(self.kernel.l,self.kernel.l))
		for i in range(self.kernel.l):
			for j in range(self.kernel.l):
				c_matrix[i,j] = (self.beta[j] and self.beta[j]*self.kernel.xx[i,j]/self.kernel.l or 0)
		return abs( sum(c_matrix) - 1.0 ) < .0001

		
class input_svm(input_base):
# implements inputs using the SVM architecture

	# observation_class = observation_base
	# observation_list_class = observation_list_base
	# o = observation_list_class()		# a list of class observation instances defining the observations of this input
	# clusters = list()				# a set of cluster spaces operating on this input
		
	t_cache = {}					# caches the most recent function (as end of interval)
	kernel = kernel()				# the kernel function used for estimating functions	
		
	def estimate(self, time=None, hypotheses = None):
		
		# generate any missing functions needed by clusters using this input
		#
		# self.t_cache stores the most recent interval recorded.  If the most
		# recent observation is after the last time interval end point, compute
		# however many functions are needed to bring self.t_cache in front
		# of the most recent observation
		
		estimates = list()
		for cluster in self.clusters:
			# test predictive scope
			if time - datetime.now() > cluster.t_delta:
				break
			
			# generate any missing functions
			while not cluster.t_delta in self.t_cache.keys() or self.o[-1].t > self.t_cache[cluster.t_delta]:
				cluster.f.append( function( self.o.interval( self.t_cache[cluster.t_delta]+cluster.t_delta, cluster.t_delta), self.kernel ) )
				self.t_cache[cluster.t_delta] += cluster.t_delta
		
			# update cluster
			cluster.optimize()
		
			# generate estimate
			# NOTE: this is not using the hypotheses at all currently - some method of
			# determining the time location in intervals will be required.
			known_data = function( self.o.interval(time-cluster.t_delta,cluster.t_delta), self.kernel )
			estimates.add( cluster.infer( known_data ) )
			
		# combine estimates
		return self.aggregate( estimates, time )
	
	def aggregate(self,estimates,time):
		pass
