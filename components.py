#! /usr/bin/env python

import sys, getopt, math, datetime, os, cmath
from math import sqrt, log

import numpy
from numpy import *
from scipy import spatial
from networkx import *
from pylab import *
from svm import *
	
class data_vector:
	def __init__(self,data,*args,**kargs):
		self.SV_matrix = None
		self.cluster = None
		self.data = array(data)
	
class support_vector(data_vector):
	def __init__(self,beta,*args,**kargs):
		self.beta = beta
		data_vector.__init__(self,*args,**kargs)
		
class inference_module:
	def __init__(self,data=list(),gamma=None,nu=1e-10):
		self.param = svm_parameter(svm_type=ONE_CLASS, kernel_type = RBF)
		self.svm = None
		self.data = data
		self.SV = list()
		self.kernel = None
		self.rho = None
		self.clusters = list()

		if nu:
			self.param.nu = nu
		if gamma:
			self.param.gamma = gamma
		if self.data:
			self._compute()
	
	def __iadd__(self, points):
		# overloaded '+=', used for adding a vector list to the module's data
		# 
		# @param points		A LIST of observation vectors (not a single ovservation)
		self.data += points
	
	def induce(self,vector,append=True):
		# Induce from vector
		# Creates an abstract representation of the given point using inductive inference based on similarity to previously observed points
		# 
		# @param point		The observation vector for which to create an abstract representation
		# @param append		Adds the point to the module's data set if True
		
		if append:
			self.data.append(vector)
			
		self._check_recompute()
		
		#point.SV_array = self._anorm( vector.data, array([SV.data for SV in self.SV]) )
		vector.SV_array = array( spatial.distance.cdist( [vector.data,], [SV.data for SV in self.SV] ) )
		
		if self.svm.predict(vector.data) > 0:
			vector.cluster = self.SV[ vector.SV_array.argmin() ].cluster
		
		return vector
		
	def deduce(self,abstraction):
		# Deduce a Probability Distribution Vector 
		# Creates a Probability Distribution Vector from the given abstract representation using deductive inference based on previously observed vectors.
		pass
		
		
	def _check_recompute(self):
		if self.data and not self.SV:
			self._compute()
		
	def _compute(self,path='output.svm'):
		if not self.param.gamma:
			self.param.gamma = 1/spatial.distance.pdist(self.data, 'euclidean').max()
		
		self.svm = svm_model(svm_problem( range(len(self.data)), self.data ),self.param)
		self.svm.save(path)
		
		parse_file = file(path,'r')
		lines = parse_file.readlines()
		self.rho = float( lines[5].split(' ')[1] )

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
			
		self._boundaries()
		print "gamma=%s" % self.param.gamma
		print "nu=%s" % self.param.nu
			
	def _anorm(self,x,y):
		return numpy.sqrt(((x - y)**2).T.sum(0))
		
	def _norm(self,x,y):
		return sqrt(((x - y)**2).sum())
		
	def _K(self,norm):
		return exp(-(norm**2)/self.gamma)
		
	def _boundaries(self):
	# determine cluster boundaries and add any clusters which do not yet exist
	
		# Construct SV adjacency matrix
		d = len(self.SV)
		N = spatial.distance.squareform( spatial.distance.pdist( [SV.data for SV in self.SV] , 'euclidean' ) )
		#N = zeros([d,d])
		#for i in range(d):
		#	for j in range(i,d):
		#		if i==j:
		#			N[i,j] = -1
		#		else:
		#			val = self._norm(self.SV[i].data, self.SV[j].data)
		#			N[i,j] = val
		#			N[j,i] = val
				
		# Calculate R^2
		# R^2(x) = K_(x,x) - 2 sum_j{\beta_j K(x_j,x)} + sum_{i,j}{\beta_i \beta_j K(x_i,x_j)
		#M = self.K( N )
		#betas = array( [ [ sv.beta for sv in self.SV] ] )
		#R2 = complex( 1 - 2* ( dot( M[:1:], betas.T ) ) + ( dot( dot(betas, M), betas.T) )[0,0] )
		
		R2 = self.rho
		
		# Calculate Z
		#Z = \sqrt{ -\frac{ln( \sqrt{ 1-R^2} )}{q} }
		Z =  abs( cmath.sqrt( -1* cmath.log( cmath.sqrt( 1- ( R2 ** 2 ) ) )  / self.param.gamma ) )

		# Assign SV's to clusters
		G = Graph( N <= Z )
		clusters = connected_component_subgraphs(G)
		for i in range(len(clusters)):
			self.clusters.append(list())
			for j in clusters[i].nodes():
				self.SV[j].SV_array = 1/N[j]
				self.SV[j].cluster = i
				self.clusters[i].append(self.SV[j])
				

			
