#! /usr/bin/env python

import sys, getopt, math, datetime, os
from math import sqrt, log

from numpy import *

from cvxmod import *
from cvxmod.atoms import quadform
from cvxmod.sets import probsimp

def parseSVM(file):
	# return a tuple containing all SV and BSV found by the SVM
	
class kernel:
	def __init__(self,C=1,q=None):
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
		
		if not self.q:
			self.q = 1/max(self.xx_norm)
			print 'q set to %s' % self.q			
			
		self.xx = self.calc(self.xx_norm)
		
		return self

	def calc(self,norm):
	# returns the gaussian distance between two functions after calculating the strict distance
		s=norm.size
		new =exp( -self.q * norm )
		return matrix(new,s)
		
	def norm(self,x,y):
		return sqrt(((x-y)**2).sum())
			
class inference_module:
	
	def __init__(self,C=1.0,*args,**kargs):
		self.kernel = kernel(C=1)
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
		sTime = datetime.datetime.now()
		
		self.kernel.load(data)
		
		d = self.kernel.l
		
		# construct objective functions
		P = param('P',d,d)
		P.psd = True
		P.semidefinite = True
		
		q = param('q',d,1)
		A = param('A',1,d)
		x = optvar('x', d,1)
		x.pos = True
		
		P.value = self.kernel.xx
		q.value =P.value[:d]
		A.value = matrix(1.0,(1,d))
		
		#print self.kernel.xx_norm
		#p = problem( minimize(   tp(q)*x + quadform(x,P) ), [x <= self.kernel.C, A*x == 1.0])
		p = problem( minimize( quadform(x,P) + tp(q)*x ), [x <= self.kernel.C, A*x == 1.0])
		p.solve(True)
		self.beta = array(x.value)
		
		print 'coef sum to %s' % self.beta.sum()
		print '%s coef larger than C (%s), %s BSV' % ( (self.beta > self.kernel.C ).sum(), self.kernel.C, (self.beta - self.kernel.C < 1e-8).sum() )
		
		
		# determine clusters
		self.boundaries()
		
		svCount = ( self.beta > 1e-5 ).sum()
		stray = svCount
		for c in self.clusters:
			stray -= len(c)
			
		print "Clusters generated in %ss using C=%s, q=%s" % ((datetime.datetime.now()-sTime).seconds,self.kernel.C, self.kernel.q)
		print "%s SV's out of %s total observations" % (svCount, self.kernel.l)
		print "%s clusters and %s stray SV's found" % (len(self.clusters), stray)
		
		# release resources from kernel cache
		#self.kernel.flush()
		
	def boundaries(self):
	# determine cluster boundaries and add any clusters which do not yet exist
		# create SV mask
		SV = ( 1e-5 < self.beta ) * ( self.beta < self.kernel.C )
		
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
		M = ( array( self.kernel.xx_norm) < Z) * SV * SV.T
		
		# Assign SV's to clusters
		for i in range(self.kernel.l):
			
			# if the point is an SV and has not been added to a cluster yet
			if SV[i] and (not i or not M[i,:i-1].sum() ):
				l=[self.kernel.x[i],]
				for j in range(i,self.kernel.l):
					if M[i,j]:
						l.append(self.kernel.x[j])
				self.clusters.append(l)

