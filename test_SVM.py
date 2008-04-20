#! /usr/bin/env python

import py.test

import CPDF_SVM as svm

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
	pass
	
def test_function():
	pass
	
def test_input():
	pass
	
