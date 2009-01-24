#! /usr/bin/env python

import sys, getopt, math, datetime, os
from random import gauss

from santa_fe import getData

from components import *

from numpy import *
from pylab import *

_Functions = ['run']
	
def run():
	print "Starting"
	
	print "Loading Dataset"
	# Retrieve dataset
	#data = getData('B1.dat')[:100]
	data = list()
	for i in range(100):
		data.append( array( [gauss(2.0,.1), gauss(0.0,.1) ]) )
	for i in range(100):
		data.append( array( [gauss(0.0,.1), gauss(2.0,.1) ]) )
		
	mod = inference_module(data,gamma=10, nu=.1)
	
	clusters = [ [] , ]*(len(mod.clusters))
	bsv = list()
	colors = ['b','r','g','c','m','y','w']

	for point in data:
		vector = mod.induce( data_vector(point),False)
		
		if vector.cluster != None:
			if not clusters[vector.cluster]:
				clusters[vector.cluster] = [vector,]
			else:
				clusters[vector.cluster].append(vector)
		else:
			bsv.append(vector)
				
	print "Formatting Results"
	for cluster in clusters:
		if cluster:
			scatter( [point.data[0] for point in cluster], [point.data[1] for point in cluster],s=10,  c=colors[cluster[0].cluster % 7])
		
	for cluster in mod.clusters:
		scatter( [SV.data[0] for SV in cluster],  [SV.data[1] for SV in cluster],  s=30,  c=colors[cluster[0].cluster % 7],  label="Cluster %s" % cluster[0].cluster )
			
	if bsv:
		scatter( [v.data[0] for v in bsv], [v.data[1] for v in bsv], s=10, marker="x" )
			
	title('%s SV in %s clusters from %s points' % (len(mod.SV),len(mod.clusters),len(data)))
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
