#! /usr/bin/env python

import py.test

import CPDF_SVM as svm
from system_2_base import observation_base, observation_list_base

import sys, getopt, math, datetime, os
from math import sqrt, sin

from numpy import *
from pylab import plot,bar,show,legend,title,xlabel,ylabel,axis

from cvxopt.base import *
from cvxopt.blas import dot 
from cvxopt.solvers import qp

from cvxopt import solvers
solvers.options['show_progress'] = False

def test_sign():
	assert svm.sign(1)
	assert not svm.sign(0)
	assert svm.sign( array([1,2,3]) )
	assert not svm.sign( array([0,1,2]) )
	assert not svm.sign( array([-1,1,2]) )
	assert not svm.sign( array([[1,2,3],[4,-5,6]]) )
	
def test_kernel():
	K = svm.kernel(gamma=.05,sigma_q=.25)
	l = observation_list_base()
	l.append( observation_base(val=5,t=datetime.datetime.now() ) )
	l.append( observation_base(val=6,t=datetime.datetime.now() ) )
	l.append( observation_base(val=7,t=datetime.datetime.now() ) )
	l.append( observation_base(val=8,t=datetime.datetime.now() ) )
	l.append( observation_base(val=9,t=datetime.datetime.now() ) )
	l.append( observation_base(val=9,t=datetime.datetime.now() ) )
	
	K.load(l)
	
	assert K.x
	assert K.y
	assert K.xx
	assert K.yy
	assert K.intg
	assert K.l == 6
	assert K.xx.size == (K.l,K.l)
	assert K.yy.size == (K.l,K.l)
	assert K.intg.size == (K.l,K.l)
	assert K.xy(1,2)
	
def test_function():
	K = svm.kernel(gamma=.05,sigma_q=.25)
	l = observation_list_base()
	test_t = datetime.datetime.now()
	l.append( observation_base(val=5,t=test_t ) )
	l.append( observation_base(val=6,t=datetime.datetime.now() ) )
	l.append( observation_base(val=7,t=datetime.datetime.now() ) )
	l.append( observation_base(val=8,t=datetime.datetime.now() ) )
	l.append( observation_base(val=9,t=datetime.datetime.now() ) )
	l.append( observation_base(val=9,t=datetime.datetime.now() ) )
	
	F = svm.function_svm(l,K)
	assert F.equality_check(), F.equality_check()
	assert abs( F.reg(test_t) - 5) < .001
	assert len(F.SV) <= 6
	
def test_input():
	pass
	
