#! /usr/bin/env python

import sys, getopt, math, datetime, os, cmath
from math import sqrt, log

import numpy
from numpy import *
from networkx import *

	
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
		
		self.clusters = list()
		
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
			
	def anorm(self,x,y):
		return numpy.sqrt(((x - y)**2).T.sum(0))
		
	def norm(self,x,y):
		return sqrt(((x - y)**2).sum())
		
	def K(self,norm):
		return exp(-self.gamma*(norm**2))
		
	def classify(self,point):
		point.SV_array = self.anorm( point.data, array([SV.data for SV in self.SV]) )
		point.cluster = self.SV[ point.SV_array.argmin() ].cluster
		
		return point
		
	def boundaries(self):
	# determine cluster boundaries and add any clusters which do not yet exist
	
		# Construct SV adjacency matrix
		d = len(self.SV)
		N = zeros([d,d])
		for i in range(d):
			for j in range(i,d):
				if i==j:
					N[i,j] = -1
				else:
					val = self.norm(self.SV[i].data, self.SV[j].data)
					N[i,j] = val
					N[j,i] = val
				
		# Calculate R^2
		# R^2(x) = K_(x,x) - 2 sum_j{\beta_j K(x_j,x)} + sum_{i,j}{\beta_i \beta_j K(x_i,x_j)
		betas = array( [ [ sv.beta for sv in self.SV] ] )
		R2 = complex( 1 - 2* ( dot( N[:1:], betas.T ) ) + ( dot( dot(betas, N), betas.T) )[0,0] )
		
		# Calculate Z
		#Z = \sqrt{ -\frac{ln( \sqrt{ 1-R^2} )}{q} }
		Z =  abs( cmath.sqrt( -1* cmath.log( cmath.sqrt( 1- ( R2 ** 2 ) ) )  / self.gamma ) )
		
		# Determine a 'good' gamma starting point
		self.gamma_start = 1/N.max()
		G = Graph( N <= Z )
		
		# Assign SV's to clusters
		clusters = connected_component_subgraphs(G)
		for i in range(len(clusters)):
			self.clusters.append(list())
			for j in clusters[i].nodes():
				self.SV[j].SV_array = 1/N[j]
				self.SV[j].cluster = i
				self.clusters[i].append(self.SV[j])
				
class data_vector:
	def __init__(self,data,*args,**kargs):
		self.SV_matrix = None
		self.cluster = None
		self.data = array(data)
	
class support_vector(data_vector):
	def __init__(self,beta,*args,**kargs):
		self.beta = beta
		data_vector.__init__(self,*args,**kargs)
			
