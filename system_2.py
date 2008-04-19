

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

class sys_2:
# top level interface for the processing architecture
# CPDF:		algorithms for determining optimal Conditional Probability Distribution Functions
# cluster:		algorithms for clustering
# APDF:		algorithms for aggregating Probability Distribution Functions
# age_function:	algorithm to determine the kill time for a function
# t_delta_init:	the t_delta value to use for new clusters on new layers - the minimum t_delta to be used by the system
	
	CPDF = None	
	cluster = None
	APDF = None
	age_function = None
	t_delta_init = None
	
	layers = list()		# the layers defined for this system
	
	def __init__(self,CPDF,cluster,APDF,age_function,t_delta_init):
		self.CPDF = CPDF
		self.cluster = cluster
		self.APDF = ADF
		self.age_function = age_function
		self.t_delta_init = t_delta_init
		
		self.layers.append( layer(sys=self) )
		

class layer:
# The set of inputs and cluster spaces which define a processing layer

	sys = None					# the system instance this layer belongs to
	cluster_spaces = list()		# cluster spaces for this layer
	inputs = list()				# inputs for this layer
	
	def __init__(self, sys):
		self.sys = sys
		self.add_cluster_space()
		
	def add_input(self):
		i = input()
		self.inputs.append(i)
		return i
	
	def add_cluster_space(self, t_delta = sys.t_delta_init ):
		self.cluster_spaces.append( cluster_space(t_delta) )

class cluster_space:
# a cluster of functions
	
	cluster = None		# clustering algorithm to use
	
	t_delta = None		# the time interval length for this cluser
	p = list()			# cluster radii in feature space
	q = None			# soft margin
	
	functions = list()	# list of functions to cluster
	C = list()			# list of defined clusters	
	
	def __init__(self, cluster, t_delta=timedelta(seconds=1)):
		self.cluster = cluster
		self.t_delta = t_delta
		
	def optimize(self):
	# determines the optimal weights for clustering the functions
	# defined in self.f and updates the clusters defined over the data
		self.cluster.optimize(self.functions)
		
	def infer(self, CPDF, time):
	# returns a PDF at the given time based on proximity of *complete* intervals
	# to the *partial* interval (or hypothesis) given by CPDF for *each* p value
	# specified for this cluster space
		pass
		
class cluster:
# representation of a cluster in some space

	output = None		# The input at a higher level which this cluster maps to
	f = list()			# the SV functions defining the edge of the cluster
	beta = array()		# the SV weights
	t_delta = None		# the time delta for this cluster
	
	def __init__(self):
		pass
		
class input:
# provides a representation of a source of information to the system.
# inputs take single scalar values and a timestamp.  Each component of a
# multi-dimensional source of information should have its own input instance
	
	o = observation_list()	# a list of class observation instances defining the observations of this input
	f = list()				# a set of all functions defined over the observations for this input
	k_cache = array()		# an array containing the kernel distances between observations
	t_cache = array()		# an array containing the times at which functions have been determined 			NOTE: this only really needs to be a counter
	clusters = list()		# a set of clusters operating on this input
	
	def __init__(self):
		self.t_cache = time.now()
	
	def optimize(self,t_delta,time=None):
	# computes a function to describe the data over the time interval starting at time
	# and ending at time+t_delta.  If time not provided, computes a function over the 
	# time interval ending now, and starting now-t_delta
		
		# retrieve data
		beta_previous = f[-1].beta
		
		# construct constraints
		
		# optimize solution
		
		# construct function
		f = function()
		self.f.append(f)
		return f
		
	def add(self, val, t=None):
	# adds an observation to the estimate
	# if no time is specified the current system time is used
	
		self.o.append( observation(val) )
		
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
		return APDF.max_prob( estimates, time )
		
		
class function:
# a function describing the behavior of an input over a specific observed time interval
	
	alpha = None		# the function estimation parameters
	kill = None			# the kill time of the function
	
	_function_d = None	# list of computed distances to different functions
	
	def __init__(self,data=None,*args,**kargs):
		if data:
			self.optimize(data, *args, **kargs)
		
	def __sub__(self,a):
	# overloads the subtract function to compute distance between two functions or a function and a cluster
		if isinstance(a, function):
			pass
		elif isinstance(a, cluster):
			pass
		
	def optimize(self,data,CPDF,*args,**kargs):
	# optimizes data using the specified Conditional Probability Distribution Function estimator
		self.alpha = CPDF.optimize(data,*args,**kargs)
		
	def reg(self,t):
	# evaluates the most function at time t
		return self.alpha.reg(t)
		
	def den(self,t):
	# returns a probability density over the range of the function at time t
		return self.alpha.den(t)
		
class observation_list(list):
# provides a container for a series of observations

	def __init__(self):
		pass
	
	def interval(self,t_start, t_delta):
	# returns a list of observations which occur between t_start and t_start + t_delta and have not yet been killed
		pass
		
	def __append__(self,*arg,**karg):
	# overloads the add operation to keep the list sorted by time
		super(self,list).__append__(*arg,**karg)
		self.sort(sort_time)
		
class observation(list):
# provides a representation of a single piece of data at a set time

	def __init__(self, val, t=None):
		super(self,list).__init__()
		
		self[0] = t and t or time.now()
		self[1] = val
	
	def __getattr__(self,value):
		if value == 'val':
			return self[1]
		elif value == 't':
			return self[0]
