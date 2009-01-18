#! /usr/bin/env python

import sys, getopt, math, datetime, os
from math import sqrt, sin

from numpy import *
from pylab import *

from cvxopt.base import *
from cvxopt.blas import dot 
from cvxopt.solvers import qp

from cvxopt import solvers
solvers.options['show_progress'] = False

from santa_fe import getData
from components import inference_module

_Functions = ['run']
	
def run():
	print "Starting"
	
	print "Loading Dataset"
	# Retrieve dataset
	data = getData('B1.dat')[:40]
	#data = array([sin(i/4.) for i in range(33)])
	
	# test on training data
	mod = inference_module()
	mod.optimize(data)
	
	
	print "Displaying Results"
	x = list()
	y = list()
	
	for point in data:
		x.append(point[0])
		y.append(point[1])
	scatter(x,y,marker="+",label="input data")
	
	for i in len(mod.clusters):
		SVx = list()
		SVy = list()
		for point in clusters[i]:
			SVx.append(point[0])
			SVy.append(point[1])
			scatter(x,y,label="Cluster %s" % i)
	legend()
	show()

def help():
	print __doc__
	return 0
	
def process(arg='run'):
	if arg in _Functions:
		globals()[arg]()
	class Usage(Exception):    def __init__(self, msg):        self.msg = msgdef main(argv=None):	if argv is None:		argv = sys.argv	try:		try:			opts, args = getopt.getopt(sys.argv[1:], "hl:d:", ["help","list=","database="])		except getopt.error, msg:			raise Usage(msg)
				# process options		for o, a in opts:			if o in ("-h", "--help"):
				for f in _Functions:
					if f in args:
						apply(f,(opts,args))
						return 0				help()
				# process arguments		for arg in args:			process(arg) # process() is defined elsewhere
			
	except Usage, err:		print >>sys.stderr, err.msg		print >>sys.stderr, "for help use --help"		return 2if __name__ == "__main__":	sys.exit(main())
