#! /usr/bin/env python

import sys, getopt, math, datetime, os
from math import sqrt, log

from numpy import *

from cvxmod import *
from cvxmod.atoms import quadform
from cvxmod.sets import probsimp

def parseSVM(file):
	# return a tuple containing all SV and BSV found by the SVM
	
	
class module:
	def __init__(self,path):
		parse_file = file(path,'r')
		data = list()
		# parse file
		lines = parse_file.readlines()
		
		self.SV = list()
		self.BSV = list()
		self.kernel = lines[1].split(' ')[1]
		self.gamma = float( lines[2].split(' ')[1] )
		self.rho = float( lines[5].split(' ')[1] )
		
		self.gamma_start = None
		
		for line in lines[7:]
			target = self.SV
			text = line.split(' ')
			if text[0] == 'BSV':
				target = self.BSV
			beta = text[0]
			# NOTE: this is NOT using sparse datasets - each observation needs to be fully defined
			data = list()
			for value in text[1:]:
				v  = value.split(':')
				data.append( float(v[1]) )
			target.append( support_vector(beta,data) )
			
		self.boundaries()
			
	def get_SV(self):
		for sv in self.SV:
			print sv
			
	def get_BSV(self):
		for bsv in self.BSV:
			print bsv
			
	def norm(self,x,y):
		return sqrt(((x - y)**2).sum())
		
	def classify(self,point):
		point.SV_array = empty(len(self.SV))
		nearest = None
		max = 0
		for i in len(self.SV):
			point.SV_array[i] = self.norm( point, self.SV[i] )
			if point.SV_array[i] > max:
				nearest = self.SV[i]
		point.SV_array *= nearest.SV_matrix
		
		#NOTE: this should probably be checking for BSV...
		point.cluster = nearest.cluster
		
		return point
		
		
	def boundaries(self):
	# determine cluster boundaries and add any clusters which do not yet exist
	
		# Calculate Z
		#Z = \sqrt{ -\frac{ln( \sqrt{ 1-R^2} )}{q} }
		Z = sqrt( -1* log( sqrt( 1- self.rho ** 2 ) )  / self.gamma )
		
		# Construct SV adjacency matrix
		d = len(self.SV)
		M = zeros(d,d)
		for i in range(d):
			for j in range(i,d):
				val = self.norm(self.SV[i].data, self.SV[j].data)
				M[i,j] = val
				M[j,i] = val
		
		# Determine a 'good' gamma starting point
		self.gamma_start = 1/M.max()
		M = M < Z
		
		# Assign SV's to clusters
		#NOTE: THIS DOES NOT WORK!!! it's only assiging adjacency; it isn't tracing the entire perimeter
		for i in range(d):
			# Set SV[i]'s inhibition matrix to the inverse of the norm, reduced by the intra-cluster inhibition factor
			self.SV[i].SV_array = 1/M[i]*self.inhibition
			
			# if the point is an SV and has not been added to a cluster yet
			if not i or not M[i,:i-1].sum():
				self.SV[i].cluster = i
				for j in range(i,d):
					if M[i,j]:
						self.SV[j].cluster = i
						
						# set SV[i]'s inhibition matrix value for SV[j] to their norm (since they're in the same cluster)
						self.SV[i].SV_array[j] *= self.inhibition
	
class data_vector(array):
	def __init__(self,data,*args,**kargs):
		self.SV_matrix = None
		self.cluster = None
		array.__init__(self,data,*args,**kargs)
	
class support_vector(data_vector):
	def __init__(self,beta,*args,**kargs):
		self.beta = beta
		data_vector.__init__(self,*args,**kargs)
			
