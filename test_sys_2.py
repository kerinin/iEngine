#! /usr/bin/env python

from system_2_base import *
from cluster_SVC import *
from CPDF_SVM import *
from system_2 import *

def test_initialization_base:
	assert sys_2_base()
	assert layer_base()
	assert cluster_base()
	assert cluster_space_base()
	assert input_base()
	assert function_base()
	assert observation_list_base()
	assert observation_base()

def test_initialization_SVC:
	assert cluster_space_svc()

def test_initialization_SVM:
	assert input_base_svm()
	assert function_base_svm()

def test_initialization:
	assert sys_2()
	assert layer()
	assert cluster()
	assert cluster_space()
	assert input()
	assert function()
	assert observation_list()
	assert observation()

def test_overloading:
	assert sys_2().layer_class != sys_2_base().layer_class
	assert layer().input_class != layer_base().input_class
	assert layer().cluster_space_class != layer_base().cluster_space_class
	assert input().observation_class != input_base().observation_class
		

