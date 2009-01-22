#! /usr/bin/env python

import sys, getopt, math, datetime, os
from math import sqrt, log

from numpy import *

	
class inference_module:
	def __init__(self,path,inhibition=2.0):
		parse_file = file(path,'r')
		data = list()
		# parse file
		lines = parse_file.readlines()
		
		self.SV = list()
		self.kernel = lines[1].split(' ')[1]
		self.gamma = float( lines[2].split(' ')[1] )
		self.rho = float( lines[5].split(' ')[1] )
		
		self.gamma_start = None
		self.inhibition = inhibition
		
		self.cluster_count = None
		
		for line in lines[7:]:
			text = line.split(' ')
			beta = float( text[0] )
			# NOTE: this is NOT using sparse datasets - each observation needs to be fully defined
			data = list()
			try:
				for value in text[1:]:
					v  = value.split(':')
					data.append( float(v[1]) )
			except IndexError:
				pass
			support_vector(beta,data)
			self.SV.append( support_vector(beta,data) )
			
		self.boundaries()
			
	def get_SV(self):
		for sv in self.SV:
			print sv
			
	def norm(self,x,y):
		return sqrt(((x - y)**2).sum())
		
	def K(self,norm):
		return exp(-self.gamma*(norm**2))
		
	def classify(self,point):
		point.SV_array = empty(len(self.SV))
		nearest = None
		min = self.norm(point.data,self.SV[0].data)
		
		for i in range(len(self.SV)):
			point.SV_array[i] = self.norm( point.data, self.SV[i].data )
			if point.SV_array[i] < min:
				min = point.SV_array[i]
				nearest = self.SV[i]
		if nearest:
			point.SV_array *= nearest.SV_array
			point.cluster = nearest.cluster
		return point
		
		
	def boundaries(self):
	# determine cluster boundaries and add any clusters which do not yet exist
	
		# Construct SV adjacency matrix
		d = len(self.SV)
		N = zeros([d,d])
		for i in range(d):
			for j in range(i,d):
				val = self.norm(self.SV[i].data, self.SV[j].data)
				N[i,j] = val
				N[j,i] = val
				
		# Calculate R^2
		# R^2(x) = K_(x,x) - 2 sum_j{\beta_j K(x_j,x)} + sum_{i,j}{\beta_i \beta_j K(x_i,x_j)
		betas = array( [ [ sv.beta for sv in self.SV] ] )
		
		R2 = self.K(N[0,0]) - 2* ( dot( betas.T, N[:1:] ) ) + ( dot( dot(betas, N), betas.T) )
		
		#NOTE: the problem here is that the arrays are 1-d...
				
		# Calculate Z
		#Z = \sqrt{ -\frac{ln( \sqrt{ 1-R^2} )}{q} }
		Z = sqrt( -1* log( sqrt( 1- R2 ** 2 ) )  / self.gamma )
		
		# Determine a 'good' gamma starting point
		self.gamma_start = 1/N.max()
		M = N < Z
		
		# Assign SV's to clusters
		self.cluster_count = 0
		def r(base,offset,stack):
			for j in range(offset,d):	
				if not self.SV[j].cluster and M[offset,j]:
					self.SV[j].cluster = self.cluster_count
					
					# set SV[i]'s inhibition matrix value for SV[j] to their norm (since they're in the same cluster)
					self.SV[base].SV_array[j] *= self.inhibition	
			while len(stack):
				r(base,stack.pop(0),stack)
		for i in range(d):
			# Set SV[i]'s inhibition matrix to the inverse of the norm, reduced by the intra-cluster inhibition factor
			self.SV[i].SV_array = 1/N[i]*self.inhibition

			if not self.SV[i].cluster:
				self.SV[i].cluster = self.cluster_count
				stack = list()
				r(i,i,stack)
				self.cluster_count += 1
				
class data_vector:
	def __init__(self,data,*args,**kargs):
		self.SV_matrix = None
		self.cluster = None
		self.data = array(data)
	
class support_vector(data_vector):
	def __init__(self,beta,*args,**kargs):
		self.beta = beta
		data_vector.__init__(self,*args,**kargs)
			
