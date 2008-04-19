#! /usr/bin/env python

from system_2_base import cluster_space_base, cluster_base

class cluster_space_svc(cluster_space_base):
	# t_delta = None		# the time interval length for this cluser
	# f = list()		# list of functions to cluster
	# C = list()		# list of defined clusters

	p = list()			# cluster radii in feature space
	q = None			# soft margin

	def optimize(self):
	# determines the optimal weights for clustering the functions
	# defined in self.f and updates the clusters defined over the data
		raise StandardError, 'This function not implemented'
		
	def infer(self, CPDF, time):
	# returns a PDF at the given time based on proximity of *complete* intervals
	# to the *partial* interval (or hypothesis) given by CPDF for *each* p value
	# specified for this cluster space
		raise StandardError, 'This function not implemented'

class cluster_svc(cluster_base):

	# output = None		# The input at a higher level which this cluster maps to
	# t_delta = None		# the time delta for this cluster

	f = list()			# the SV functions defining the edge of the cluster
	beta = list()		# the SV weights
