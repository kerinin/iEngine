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
	# f = list()				# a set of all functions defined over the observations for this input
	# clusters = list()			# a set of clusters operating on this inpu

	k_cache = list()		# an array containing the kernel distances between observations
	t_cache = list()		# an array containing the times at which functions have been determined 			NOTE: this only really needs to be a counter

	def optimize(self,t_delta,time=None):
	# computes a function to describe the data over the time interval starting at time
	# and ending at time+t_delta.  If time not provided, computes a function over the 
	# time interval ending now, and starting now-t_delta
		
		# retrieve data
		beta_previous = self.f[-1].beta
		
		# construct constraints
		
		# optimize solution
		
		# construct function
		f = function()
		self.f.append(f)
		return f
		
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

	_function_d = None	# list of computed distances to different functions
	
	def __sub__(self,a):
	# overloads the subtract function to compute distance between two functions or a function and a cluster
		raise StandardError, 'This function not implemented'
		
	def optimize(self,data,CPDF,*args,**kargs):
	# optimizes data using the specified Conditional Probability Distribution Function estimator
		raise StandardError, 'This function not implemented'
		
	def reg(self,t):
	# evaluates the most function at time t
		raise StandardError, 'This function not implemented'
		
	def den(self,t):
	# returns a probability density over the range of the function at time t
		raise StandardError, 'This function not implemented'
