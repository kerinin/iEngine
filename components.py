#! /usr/bin/env python

import sys, getopt, math, datetime, os
from math import sqrt, log

from numpy import *

from cvxopt.base import *
#from cvxopt.blas import dot 
from cvxopt.solvers import qp

from cvxopt import solvers
solvers.options['show_progress'] = False

class kernel:
	def __init__(self,C,q):
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
		
		# calculate x distances
		#for i in range(self.l):
		#	for j in range(i,self.l):
		#		val = self.calc(self.xx_norm[i,j])
		#		self.xx[i,j] = val
		#		self.xx[j,i] = val
		self.xx = self.calc(self.xx_norm)
		
		return self

	def calc(self,norm):
	# returns the gaussian distance between two functions after calculating the strict distance
		return exp( -1 * self.q * (norm ** 2) )
		
	def norm(self,x,y):
		return sqrt( (x-y).sum() ** 2 )
			
class inference_module:
	
	def __init__(self,C=1.0,q=5e-6,*args,**kargs):
		self.kernel = kernel(C,q)
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
		
		d = self.kernel.l
		
		# construct objective functions
		P = self.kernel.xx
		q = self.kernel.xx[:d] * -1.0
		
		# construct equality constraints
		A = matrix( 1.0, (1,d) )
		b = matrix(1.0)
	
		# construct inequality constraints
		#G = matrix(1.0/d, (d,d))
		#h = matrix(self.kernel.C, (d,1))
		G = matrix(1.0/self.kernel.C,(1,d) )
		h = matrix(0.0)
		
		# optimize and set variables
		optimized = qp(P, q,G=G,h=h,A=A, b=b)
		self.f = self.kernel.x
		self.beta = array(optimized['x'])
		
		print self.beta
		
		# determine sphere radius
		# the sphere radius is the function we're maximizing
		# \sum_i \beta_i K(x_i, x_i) - \sum_{i,j} \beta_i \beta_j K(x_i, x_j)
		#self.R_2 = self.beta.T * P * self.beta + self.beta[0::self.kernel.l] * q
		
		# determine clusters
		# self.boundaries()
		
		# release resources from kernel cache
		self.kernel.flush()
		
	def boundaries(self):
	# determine cluster boundaries and add any clusters which do not yet exist
		# create SV mask
		SV = ( self.kernel.C/10000000 < self.beta ) * ( self.beta <= self.kernel.C )
		
		# Calculate R
		#R^2(x) = K(x,x) - 2 \sum_j \beta_j K(x_j,x) + \sum_{i,j} \beta_i \beta_j K(x_i,x_j)
		R = None
		for i in range(self.kernel.l):
			if SV[i,0]:
				R = sqrt( self.kernel.xx[i,i] - 2 * ( self.beta * self.kernel.xx[i::self.kernel.l] ).sum() + ( self.beta.T * self.kernel.xx * self.beta ).sum() )
				break
		
		# Calculate Z
		#Z = \sqrt{ -\frac{ln( \sqrt{ 1-R^2} )}{q} }
		Z = sqrt( -1* log( sqrt( 1- R ** 2 ) )  / self.kernel.q )
		
		# Construct SV adjacency matrix
		M = (SV*self.kernel.xx_norm) < Z
		
		# Assign SV's to clusters
		for i in range(self.kernel.l):
			# if the point is an SV and has not been added to a cluster yet
			if M[i,0] and not M[i::i].sum():
				l=[self.x[i],]
				for j in range(i,self.kernel.l):
					if M[i,j]:
						l.append(self.x[j])
				self.clusters.append(l)
