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
	kernel = None				# the kernel function used for estimating functions	
	
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

