#! /usr/bin/env python

import sys, getopt, math, datetime, os
from random import gauss

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_Functions = ['run', 'test_parzen', 'test_divergence']
	
import theano.tensor as T
from theano import function

def run():
  print "Starting"
  
  import system
  m = system.model()
  
  print "compiled for GPU"
  
  xRange=[0,100]
  yRange=[-1.5,1.5]
  
  # Generate observation sequence
  oRange=[0,50]
  oStep=.5
  oN = int((oRange[1]-oRange[0]/oStep))
  oX = np.arange(oRange[0],oRange[1],oStep).astype('float32')
  
  oY = np.sin(oX/2)
  
  # Generate predictions
  pRangeX = [50,100]
  pRangeY = yRange
  pStepX = .1
  pStepY = .01
  pxN = int((pRangeX[1]-pRangeX[0])/pStepX)
  pyN = int((pRangeY[1]-pRangeY[0])/pStepY)
  P = np.dstack(np.mgrid[pRangeX[0]:pRangeX[1]:pStepX,pRangeY[0]:pRangeY[1]:pStepY]).reshape([ pxN*pyN,2])
  px = np.arange(pRangeX[0],pRangeX[1],pStepX)
  py = np.arange(pRangeY[0],pRangeY[1],pStepY)
  gamma = .1
  
  for i in range(oN):
    m.process( [oY[i]], time=oX[i])
  
  print len(m.layers)
  print m.layers[0].sequences.shape
   
  pz = (P[:,0]+P[:,1]).reshape([pxN,pyN]).T
  
  CS = plt.contour(px,py,pz,10)
  plt.clabel(CS, inline=1, fontsize=10)
  
  plt.scatter( oX, oY)
  plt.axis( [xRange[0], xRange[1], yRange[0], yRange[1]] )
  
  plt.show()
  
def test_divergence():
  print "Starting"
  
  import cs_divergence, parzen_probability
  
  print "compiled for GPU"
  
  xrange = [0,1]
  xstep = .01
  xN = int((xrange[1]-xrange[0])/xstep)
  x=np.arange(xrange[0],xrange[1],xstep).astype('float32')
  gamma = .1
  distN = 20
  baseN = 20
  
  # 5 distributions containing 5 1-d points
  distributions = np.array( [
    np.random.normal(.2, .05, distN), 
    np.random.normal(.4, .05, distN), 
    np.random.normal(.6, .05, distN),  
    np.random.normal(.8, .05, distN)
  ] ).reshape(4,distN,1).astype('float32')
  # distribution with 5 1-d points
  base = np.random.normal(.8, .05, baseN).reshape(baseN,1).astype('float32')
  
  divergences = cs_divergence.from_many(distributions, base, gamma=gamma)
  
  for i in range(4):
    ax = plt.subplot(2,2,i+1, title="Divergence: %s" % divergences[i])
    
    ax.plot(x, parzen_probability.from_many( distributions[i].reshape(1,distN,1), x.reshape(xN,1), gamma=gamma ).reshape(xN), 'b' )
    ax.plot(x, parzen_probability.from_many( base.reshape(1,baseN,1), x.reshape(xN,1), gamma=gamma ).reshape(xN), 'g--' )
    ax.axis([0,1,0,None])
  
  plt.show()
  
  
def test_parzen():
	print "Starting"
	
	import parzen_probability
	
	print "compiled for GPU"
	
	xrange = [0,1]
	yrange = [0,1]
	xstep = .01
	ystep = .01
	xN = int((xrange[1]-xrange[0])/xstep)
	yN =  int((yrange[1]-yrange[0])/ystep)
	X = np.dstack(np.mgrid[xrange[0]:xrange[1]:xstep,yrange[0]:yrange[1]:ystep]).reshape([ xN *yN,2])
	x = np.arange(xrange[0],xrange[1],xstep)
	y = np.arange(yrange[0],yrange[1],ystep)
	gamma = .1

	# [distribution][observation][dimension]
	observations = np.random.rand(1,10,2).astype('float32')
	
	# [test point][dimension]
	test_points = X.astype('float32').reshape(xN*yN,2)
	observation_label_points = observations.reshape(10,2)
	
	# NOTE: the PDF contours seem to be *displaying* properly (not being computed properly though)
	pdf = parzen_probability.from_many(observations,test_points,gamma)
	observation_probability = parzen_probability.from_many(observations,observation_label_points,gamma)
  
	z = pdf.reshape([xN,yN]).T
	sizes = observation_probability.reshape(10)
	sizes = sizes * ( 50/sizes.max() )
	
	
	CS = plt.contour(x,y,z,10)
	plt.scatter( observations[0,:,0], observations[0,:,1], sizes ) # NOTE: these points seem to be displaying properly
	plt.clabel(CS, inline=1, fontsize=10)
	plt.axis( [xrange[0],xrange[1],yrange[0],yrange[1]] )
	plt.show()
  
def help():
	print __doc__
	return 0
	
def process(arg='run'):
	if arg in _Functions:
		globals()[arg]()
	
class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(argv=None):
	if argv is None:
		argv = sys.argv
	try:
		try:
			opts, args = getopt.getopt(sys.argv[1:], "hl:d:", ["help","list=","database="])
		except getopt.error, msg:
			raise Usage(msg)
		
		# process options
		for o, a in opts:
			if o in ("-h", "--help"):
				for f in _Functions:
					if f in args:
						apply(f,(opts,args))
						return 0
				help()
		
		# process arguments
		for arg in args:
			process(arg) # process() is defined elsewhere
			
	except Usage, err:
		print >>sys.stderr, err.msg
		print >>sys.stderr, "for help use --help"
		return 2

if __name__ == "__main__":
	sys.exit(main())
