
from CPDF_SVM import *
from cluster_SVC import *

class sys_2(sys_2_base):
	layer_class = layer

class layer(layer_base):
	input_class = input
	cluster_space_class = cluster_space

class cluster_space(cluster_space_svc):
	pass

class cluster(cluster_base):
	pass

class input(input_svm):
	observation_class = observation

class function(function_svm):
	pass

class observation_list(observation_list_base):
	pass

class observation(observation_base):
	pass
