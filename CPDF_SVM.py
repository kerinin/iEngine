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

class input_svm(input_base):

	# observation_class = observation_base
	# observation_list_class = observation_list_base
	# o = observation_list_class()		# a list of class observation instances defining the observations of this input
	# clusters = list()			# a set of cluster spaces operating on this input

	def optimize(self,t_delta,time=None):
	# computes a function to describe the data over the time interval ending at time
	# and starting at time-t_delta.  If time not provided, computes a function over the 
	# time interval ending now, and starting now-t_delta
		if not time: time = datetime.now()

		# generate functions
		for cluster in self.clusters:
			data = self.o.interval(time,cluster.t_delta)
			function = function(data)
			cluster.f.append(function)
		
	def estimate(self, time=None, hypotheses = None):
	# estimates the input's value at time under the constraints that at the time/value pairs
	# in hypothesis the input has the specified values.
		
		# generate any missing functions needed by clusters using this input
		#
		# self.t_cache stores the most recent interval recorded.  If the most
		# recent observation is after the last time interval end point, compute
		# however many functions are needed to bring self.t_cache in front
		# of the most recent observation
		
		estimates = list()
		for cluster in self.clusters:
			#NOTE: this is not going to work any more			
			while self.observation_list[-1].t > self.t_cache[cluster.t_delta]:
				self.f.append( function( self.o.interval( self.t_cache[cluster.t_delta]+cluster.t_delta, cluster.t_delta) ) )
				self.t_cache[cluster.t_delta] += cluster.t_delta
		
			# update cluster
			cluster.optimize()
		
			# generate estimate
			# NOTE: this is not using the hypotheses at all currently - some method of
			# determining the time location in intervals will be required.
			estimates.add( cluster.infer( function( self.o.interval( time, cluster.t_delta ) ) ) )
			
		# combine estimates
		return self.aggregate( estimates, time )
	
	def aggregate(self,estimates,time):
		pass

class function_svm(function_base):
# a function describing the behavior of an input over a specific observed time interval

	# kill = None		# the kill time of the function

	SV = list()		# list of SV functions
	beta = list()		# list of SV multipliers
	_function_d = None	# list of computed distances to different functions
	
	def __sub__(self,a):
	# overloads the subtract function to compute distance between two functions or a function and a cluster
		raise StandardError, 'This function not implemented'
		
	def optimize(self,data,*args,**kargs):
	# optimizes data
		# get kernel values
		
		# construct objective functions
		
		# construct equality constraints

		# construct inequality constraints

		# optimize and set variables
		
	def reg(self,t):
	# evaluates the most likely value of the function at time t
		ret = zeros(self.kernel.n)
		for i in range(self.kernel.l):
			ret += self.kernel.y[i]*self.beta[i]*self.kernel._calc(x,self.kernel.x[i])
		return ret
		
	def den(self,t):
	# returns a probability density over the range of the function at time t
		raise StandardError, 'This function not implemented'
