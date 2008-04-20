#! /usr/bin/env python

# This module provides a general class heirarchy and set of methods to implement what I'm calling system 2, and attempts to make
# no assumptions about the nature of the underlying optimization algorithms.  The only specific algorithms included in this code
# are algorithms to control and optimize the evolution of the system over time.  This would include decision functions for when to
# add new cluster sets, when to add new function interval lengths, and when to add additional layers.
#
# Usage
# 1) input data
# data is inserted into each input in the system.  Data MUST be inserted in chronological order
# If data is inserted which is older than the most recent observation, it will not be included
# in the optimization procedure
#
# 2) retrieve estimates
# inputs are queried for an estimate at a spcific time
# all optimization and clustering is done transparently at the time of a 
# data request	

from datetime import *

class observation_base(list):
# provides a representation of a single piece of data at a set time

	def __init__(self, val, t=None):
		super(list,self).__init__()
		
		self.append( t and t or datetime.now() )
		self.append( val )
	
	def __getattr__(self,value):
		if value == 'val':
			return self[1]
		elif value == 't':
			return self[0]
			
class observation_list_base(list):
# provides a container for a series of observations
	
	def interval(self,t_start, t_delta):
	# returns a list of observations which occur between t_start and t_start + t_delta and have not yet been killed
		#NOTE: this is not very efficient - the list is sorted		
		ret = list()
		for i in self:
			if i.t >= t_start and i.t <= (t_start + t_delta):
				ret.append(i)
		return ret

class function_base:
# a function describing the behavior of an input over a specific observed time interval
	
	kill = None		# the kill time of the function
	
	def __init__(self,data=None,*args,**kargs):
		if data:
			self.optimize(data, *args, **kargs)
		
	def __sub__(self,a):
	# overloads the subtract function to compute distance between two functions or a function and a cluster
		raise StandardError, 'This function not implemented'
		
	def optimize(self,data,*args,**kargs):
	# optimizes data using the specified Conditional Probability Distribution Function estimator
		raise StandardError, 'This function not implemented'
		
	def reg(self,t):
	# evaluates the most function at time t
		raise StandardError, 'This function not implemented'
		
	def den(self,t):
	# returns a probability density over the range of the function at time t
		raise StandardError, 'This function not implemented'
		
class input_base:
# provides a representation of a source of information to the system.
# inputs take single scalar values and a timestamp.  Each component of a
# multi-dimensional source of information should have its own input instance
	
	observation_class = observation_base
	observation_list_class = observation_list_base

	o = observation_list_class()		# a list of class observation instances defining the observations of this input
	clusters = list()			# a set of cluster spaces operating on this input
		
	def __init__(self):
		self.t_cache = datetime.now()
		
	def add(self, val, t=None):
	# adds an observation to the estimate
	# if no time is specified the current system time is used
	
		self.o.append( self.observation_class(val=val,t=t) )
		
	def attach(self,cluster_space):
	# attaches this input to the cluster space specified
		if not cluster in self.clusters:
			self.clusters.append(cluster_space)

	def estimate(self, time=None, hypotheses = None):
	# estimates the input's value at time under the constraints that at the time/value pairs
	# in hypothesis the input has the specified values.
		
		raise StandardError, 'This function not implemented'

	def aggregate(self,estimates,time):
	# combines multiple CPDF's into a single PDF or value

		raise StandardError, 'This function not implemented'
	
class cluster_base:
# representation of a cluster in some space

	output = None		# The input at a higher level which this cluster maps to
	t_delta = None		# the time delta for this cluster

class cluster_space_base:
# a cluster of functions
	
	t_delta = None		# the time interval length for this cluser
	
	f = list()		# list of functions to cluster
	C = list()		# list of defined clusters	
	
	def __init__(self, t_delta=timedelta(seconds=1)):
		self.t_delta = t_delta
		
	def optimize(self):
	# determines the optimal weights for clustering the functions
	# defined in self.f and updates the clusters defined over the data
		raise StandardError, 'This function not implemented'
		
	def infer(self, CPDF, time):
	# returns a PDF at the given time based on proximity of *complete* intervals
	# to the *partial* interval (or hypothesis) given by CPDF for *each* p value
	# specified for this cluster space
		raise StandardError, 'This function not implemented'

class layer_base:
# The set of inputs and cluster spaces which define a processing layer

	input_class = input_base
	cluster_space_class = cluster_space_base

	sys = None				# the system instance this layer belongs to
	cluster_spaces = list()			# cluster spaces for this layer
	inputs = list()				# inputs for this layer
	
	def __init__(self, sys):
		self.sys = sys
		self.add_cluster_space()
		
	def add_input(self):
		i = input_class()
		self.inputs.append(i)
		return i
	
	def add_cluster_space(self, t_delta = None ):
		if not t_delta:
			t_delta = self.sys.t_delta_init
		self.cluster_spaces.append( self.cluster_space_class(t_delta) )

class sys_2_base:
# top level interface for the processing architecture
# CPDF:		algorithms for determining optimal Conditional Probability Distribution Functions
# cluster:		algorithms for clustering
# APDF:		algorithms for aggregating Probability Distribution Functions
# t_delta_init:	the t_delta value to use for new clusters on new layers - the minimum t_delta to be used by the system
	
	layer_class = layer_base
	
	t_delta_init = None
	layers = list()		# the layers defined for this system
	
	def __init__(self,t_delta_init):
		self.t_delta_init = t_delta_init
		self.layers.append( self.layer_class(sys=self) )
