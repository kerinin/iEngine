#! /usr/bin/env python

from system_2_base import cluster_space_base

class cluster_space_svc(cluster_space_base):
		
	def optimize(self):
	# determines the optimal weights for clustering the functions
	# defined in self.f and updates the clusters defined over the data
		raise StandardError, 'This function not implemented'
		
	def infer(self, CPDF, time):
	# returns a PDF at the given time based on proximity of *complete* intervals
	# to the *partial* interval (or hypothesis) given by CPDF for *each* p value
	# specified for this cluster space
		raise StandardError, 'This function not implemented'
