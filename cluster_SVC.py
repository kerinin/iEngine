#! /usr/bin/env python

from system_2_base import cluster_space_base, cluster_base

class kernel:
	def __init__(self,C=1,q=.5):
		self.l = 0								# the number of data points cached so far
		
		self.C = C								# soft margin variable
		self.q = q								# gaussian kernel width 
		
	def flush(self):
		self.x = list()
		self.xx_norm = None
		self.xx = None
		
	def load(self,observations):
		# set variables
		self.flush()
		self.l = len(observations)
		
		self.x = observations
		self.xx_norm = matrix(0.0,(self.l,self.l))
		self.xx = matrix(0.0,(self.l,self.l))
		
		# calculate x norms
		for i in range(self.l):
			for j in range(i,self.l):
				val = self.norm(self.x[i],self.x[j])
				self.xx_norm[i,j] = val
				self.xx_norm[j,i] = val
				
		# calculate q_init
		self.q_init = 1 / (self.xx_norm.max() ** 2)
		
		# calculate x distances
		for i in range(self.l):
			for j in range(i,self.l):
				val = self.calc(self.xx_norm[i,j])
				self.xx[i,j] = val
				self.xx[j,i] = val
				
		return self

	def calc(self,norm):
	# returns the gaussian distance between two functions after calculating the strict distance
		return exp( -self.q_init * (norm ** 2) )
		
	def norm(self,x,y):
		return sqrt( sum( (x-y)^2 )
			
class inference_module(object):
	
	def __init__(self,kernel,*args,**kargs):
		self.kernel = kernel
		self.beta = list()		# function weights
		#self.R_2 = 0.0			# minimal hypersphere radius squared
		self.Z = None			# maximum distance btw SV's
		self.clusters = list()		# set of cluster SV's
		
	def optimize(self,data):
	# determines the optimal weights for clustering the functions
	# defined in self.f and updates the clusters defined over the data
	#
	# this optimization problem can be defined as such:
	# maximize 		\sum_i \beta_i K(x_i, x_i) - \sum_{i,j} \beta_i \beta_j K(x_i, x_j)
	# (minimize)		\sum_{i,j} \beta_i \beta_j K(x_i,x_j) - \sum_i \beta_i K(x_i,x_j)
	# subject to 		0 \le \beta_i \le C,  \sum_i \beta_i =1
	#
	# The quadratic optimization problem must be in the following form:
	# minimize (1/2)x^T P x + q^T x
	# subject to 		G x \le h,  A x = b
	# 
	# in our case, x = \beta, allowing us to derive P  and q as
	# P_{i,j} = K(x_i, x_j)
	# p_i = -K(x_i, x_i)
	#
	# we have two inequality constraints, which we'll represent as two columns in G and h
	# G[0,i] = -1		G[1,i] = 1 
	# h[0,i] = 0		h[1,i] = C
	#
	# finally, we have a single equality constraint which we formulate as
	# a[0,i] = 1
	# b = 1
	
		self.kernel.load(data)
		
		# construct objective functions
		P = self.kernel.xx
		q = self.kernel.xx[:self.kernel.l] * -1.0
		
		# construct equality constraints
		A = matrix( 1.0, (1,self.kernel.l) )
		b = matrix(1.0)
	
		# construct inequality constraints
		G = matrix( [ matrix(-1.0, (self.kernel.l,1)), matrix(1.0, (self.kernel.l,1)) ] )
		h = matrix( [ matrix(0, (self.kernel.l,1)), matrix( self.C, (self.kernel.l,1)) ] )
		
		# optimize and set variables
		optimized = qp(P, q,G=G,h=h,A=A, b=b)
		self.f = K.x
		self.beta = optimized['x']
		
		# determine sphere radius
		# the sphere radius is the function we're maximizing
		# \sum_i \beta_i K(x_i, x_i) - \sum_{i,j} \beta_i \beta_j K(x_i, x_j)
		#self.R_2 = self.beta.T * P * self.beta + self.beta[0::self.kernel.l] * q
		
		# determine clusters
		self.boundaries()
		
		# release resources from kernel cache
		self.kernel.flush()
		
	def boundaries(self):
	# determine cluster boundaries and add any clusters which do not yet exist
	
		# Calculate Z
		
		# Construct adjacency matrix
		
		# Assign SV's to clusters
		
		raise StandardError, 'This function not implemented'