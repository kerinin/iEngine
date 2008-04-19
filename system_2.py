#! /usr/bin/env python

from CPDF_SVM import *
from cluster_SVC import *
from system_2_base import *

class observation(observation_base):
	pass

class observation_list(observation_list_base):
	pass

class function(function_svm):
	pass

class input(input_svm):
	observation_class = observation
	observation_list_class = observation_list

class cluster(cluster_base):
	pass

class cluster_space(cluster_space_svc):
	pass

class layer(layer_base):
	input_class = input
	cluster_space_class = cluster_space

class sys_2(sys_2_base):
	layer_class = layer












