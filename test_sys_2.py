#! /usr/bin/env python

import py.test

from system_2_base import *
from cluster_SVC import *
from CPDF_SVM import *
from system_2 import *


from datetime import *

def test_initialization_base():
	assert sys_2_base(timedelta(seconds=1))
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
	assert sys_2(timedelta(seconds=1))
	assert cluster()
	assert input()
	assert function()
	o = observation_list()
	o.append(observation(val=5))
	assert o

def test_overloading():
	sys1 = sys_2(timedelta(seconds=1))
	sys2 = sys_2_base(timedelta(seconds=1))
	assert sys1.layer_class != sys2.layer_class
	assert layer(sys1).input_class != layer_base(sys2).input_class
	assert layer(sys1).cluster_space_class != layer_base(sys2).cluster_space_class
	assert input().observation_class != input_base().observation_class

def test_observation():
	t=datetime.now()
	o=observation(t=t,val=10)
	assert o.t == t
	assert o.val == 10

def test_observation_list():
	v1 = [datetime.now(),1]
	v2 = [datetime.now(),2]
	v3 = [datetime.now(),3]
	v4 = [datetime.now(),4]
	o_list = observation_list( [
		observation(t=v1[0],val=v1[1]),
		observation(t=v2[0],val=v1[1]),
		observation(t=v3[0],val=v1[1]),
		observation(t=v4[0],val=v1[1]) ] )
	assert len(o_list) == 4
	assert o_list.interval(v2[0],(v3[0]-v2[0])) == [observation(t=v2[0],val=v1[1]),observation(t=v3[0],val=v1[1])]

def test_input():
	i = input()
	i.add(5)
	i.add(10)
	i.add(15)
	i.add(20)
	assert len(i.o) == 4
	
	#optimization
	
	#aggregation
	
	#estimation
	
	

