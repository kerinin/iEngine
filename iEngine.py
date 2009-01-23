#! /usr/bin/env python

import sys, getopt, math, datetime, os
from math import sqrt, sin
from random import gauss

from svm import *

from numpy import *
from pylab import *

from cvxopt.base import *
from cvxopt.blas import dot 
from cvxopt.solvers import qp

from cvxopt import solvers
solvers.options['show_progress'] = False

from santa_fe import getData
from components import *

_Functions = ['run']
	
def run():
	print "Starting"
	
	print "Loading Dataset"
	# Retrieve dataset
	#data = getData('B1.dat')[:100]
	data = list()
	for i in range(60):
		data.append( array( [gauss(2.0,.1), gauss(0.0,.1) ]) )
	for i in range(60):
		data.append( array( [gauss(0.0,.1), gauss(2.0,.1) ]) )
		
	print "Optimizing Coeffiecients"
	# large gamma = small cluster
	# large nu = lots of SV
	param = svm_parameter(svm_type=ONE_CLASS, kernel_type = RBF,gamma=1,nu=.1)
	prob = svm_problem( range(120), data)
	m= svm_model(prob,param)
	m.save('output.svm')
	
	print "Parsing Output"
	mod = inference_module('output.svm')
	
	clusters = [ None , ]*(len(mod.clusters))
	colors = ['b','r','g']
	for point in data:
		vector = mod.classify( data_vector(point) )

		if not clusters[vector.cluster]:
			clusters[vector.cluster] = [vector,]
		else:
			clusters[vector.cluster].append(vector)
				
	print "Formatting Results"
	for cluster in clusters:
		scatter( [point.data[0] for point in cluster], [point.data[1] for point in cluster],s=10,  c=colors[cluster[0].cluster % 3])
		
	for cluster in mod.clusters:
		scatter( [SV.data[0] for SV in cluster],  [SV.data[1] for SV in cluster],  s=30,  c=colors[cluster[0].cluster % 3],  label="Cluster %s" % cluster[0].cluster )
			
	title('%s SV in %s clusters from %s points' % (len(mod.SV),len(mod.clusters),len(data)))
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
