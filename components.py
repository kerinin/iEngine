#! /usr/bin/env python

import sys, getopt, math, datetime, os, cmath
from math import sqrt, log

import numpy
from numpy import *
import scipy
from networkx import *
from pylab import *
frim svm import*
	
class inference_module:
	def __init__(self,param=svm_parameter(svm_type=ONE_CLASS, kernel_type = RBF),data=list()):
		self.param = param
		self.svm = None
		self.data = data
		self.SV = list()
		self.kernel = None
		self.rho = None
		self.clusters = list()
		
		if data:
			self.dump(data)
			self.compute()
	
	def compute(self,path='output.svm'):
		# calculate gamma (if not defined)
		self.gamma_start = 1/N.max()
		
		self.svm = svm_model(svm_problem( range(len(self.data)), data ),param) )
		self.svm.save(path)
		
		parse_file = file(path,'r')
		lines = parse_file.readlines()
		self.kernel = lines[1].split(' ')[1]
		self.gamma = float( lines[2].split(' ')[1] )
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
			
		self.boundaries()
			
	def get_SV(self):
		for sv in self.SV:
			print sv
			
	def anorm(self,x,y):
		return numpy.sqrt(((x - y)**2).T.sum(0))
		
	def norm(self,x,y):
		return sqrt(((x - y)**2).sum())
		
	def K(self,norm):
		return exp(-(norm**2)/self.gamma)
		
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
		#M = self.K( N )
		#betas = array( [ [ sv.beta for sv in self.SV] ] )
		#R2 = complex( 1 - 2* ( dot( M[:1:], betas.T ) ) + ( dot( dot(betas, M), betas.T) )[0,0] )
		
		R2 = self.rho
		
		# Calculate Z
		#Z = \sqrt{ -\frac{ln( \sqrt{ 1-R^2} )}{q} }
		Z =  abs( cmath.sqrt( -1* cmath.log( cmath.sqrt( 1- ( R2 ** 2 ) ) )  / self.gamma ) )

		# Assign SV's to clusters
		G = Graph( N <= Z )
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
			
