#! /usr/bin/env python

import py.test

from system_2_base import *
from cluster_SVC import *
from CPDF_SVM import *
from system_2 import *

import time
import datetime

def test_initialization_base():
	assert sys_2_base(datetime.timedelta(seconds=1))
	assert cluster_base()
	assert input_base()
	assert function_base()
	o = observation_list_base()
	o.append(observation_base(val=5))
	assert o
	

def test_initialization_SVC():
	assert cluster_space_svc()

def test_initialization_SVM():
	assert input_svm()
	assert function_svm()

def test_initialization():
	assert sys_2(datetime.timedelta(seconds=1))
	assert cluster()
	assert input()
	assert function()
	o = observation_list()
	o.append(observation(val=5))
	assert o

def test_overloading():
	sys1 = sys_2(datetime.timedelta(seconds=1))
	sys2 = sys_2_base(datetime.timedelta(seconds=1))
	assert sys1.layer_class != sys2.layer_class
	assert layer(sys1).input_class != layer_base(sys2).input_class
	assert layer(sys1).cluster_space_class != layer_base(sys2).cluster_space_class
	assert input().observation_class != input_base().observation_class
		

