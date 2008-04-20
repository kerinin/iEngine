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


class function_svm(function_base):
# implements functional estimation using the SVM architecture

	# kill = None		# the kill time of the function

	SV = list()		# list of SV functions
	beta = list()		# list of SV multipliers
	_function_d = None	# list of computed distances to different functions
		
	def __sub__(self,a):
		raise StandardError, 'This function not implemented'
		
	def optimize(self,data,kernel):
		
		# get kernel values
		
		
		# construct objective functions
		
		# construct equality constraints

		# construct inequality constraints

		# optimize and set variables
		
	def reg(self,t,kernel):
		
		ret = zeros(kernel.n)
		for i in range(kernel.l):
			ret += kernel.y[i]*self.beta[i]*kernel._calc(x,kernel.x[i])
		return ret
		
	def den(self,t,kernel):
		
		raise StandardError, 'This function not implemented'
		
class input_svm(input_base):
# implements inputs using the SVM architecture

	# observation_class = observation_base
	# observation_list_class = observation_list_base
	# o = observation_list_class()		# a list of class observation instances defining the observations of this input
	# clusters = list()				# a set of cluster spaces operating on this input
		
	t_cache = {}					# caches the most recent function (as end of interval)
	kernel = kernel()				# the kernel function used for estimating functions	
	
	def add(self,*args,**kargs):
		super(input_base,self).add(*args,**kargs)
		
		self.kernel.add( self.o[-1] )
		
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

class kernel:
	l = 0				# the number of data points cached so far
	n = 1			# the dimensionality of the data (assumed to be one in all cases here)
	
	x = list()			# the x values calculated for this kernel (time)
	y = list()			# the y values caluclated for this kernel (value)
	xx = None		# kernel values between time values
	yy = None		# kernel values between values
	intg = None		# integrals for values of y
	gamma = None	# smoothing variable
	sigma_q = None	# quantile to consider for the residual principal 
	sigma = 1		# calculated sigma value from last observation and given quantile
	
	def __init__(self,data,gamma=.5,sigma_q=.5):
		self.gamma = gamma
		self.sigma_q = sigma_q
		
	def add(self,observation):
		if not observation.t in self.x or not observation.val in self.y:
			# set variables
			self.l += 1
			
			self.intg = matrix(0.0,(self.l,self.l))
			
			if not observation.t in self.x:
				self.x.append(observation.t)
				self.xx = matrix(0.0,(self.l,self.l))
				
				#NOTE: this should only be for the new observation
				for i in range(self.l):
					for j in range(i,self.l):
						val = self._calc(self.x[i],self.x[j])
						self.xx[i,j] = val
						self.xx[j,i] = val
				# normalize
				self.xx /= (sum(self.xx)/self.l)
					
			if not observation.val in self.y:
				self.y.append(observation.val)
				self.yy = matrix(0.0,(self.l,self.l))
				
				#NOTE: this should only be for the new observation
				for i in range(self.l):
					for j in range(i,self.l):
						val = self._calc(self.y[i],self.y[j])
						self.yy[i,j] = val
						self.yy[j,i] = val
				# normalize
				self.yy /= (sum(self.yy)/self.l)
				
				#NOTE: this should only be for the new observation
				for i in range(self.l):
					for j in range(i,self.l):
						val = self.int(i,j)
						self.intg[i,j] = val
						self.intg[j,i] = val
						
			self.sigma = None #NOTE: this should be the quantile code - figure out what the hell to do
		
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
