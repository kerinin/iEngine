#! /usr/bin/env python

import sys, getopt, math, datetime, os, cmath
from random import gauss

import numpy as np
import numpy.ma as ma
import scipy
import scipy.special
import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
	
	
class engine:
	def __init__(self, estimates):
		
		self.estimates = estimates
		
	def pdf(self,X,S):
	# @param X		[Nxd] array of points for which to calculate a probability value
	# @param X		[Nxd] array of sample points for calculating entropy of existing estimates
		
		prior = hstack( [ phi_n.pdf(X) for phi_n in self.estimates ] )
		H_S = self._H( S )	
		
		return ( prior * ( self._varphi( H_S ) / self._varphi( H_S + self._H( X ) ) ) ).prod(1)
		
	def _varphi( self, delta ):
	# Probability distribution of known estimates' entropy
	#
	# NOTE: this is currently doing nothing - will need to implement an actual algorithm here later
	
		return 1/float(len(self.estimates))
		
	def _H(self,X):
	# @param X		[Nxd] array of observations
		P = [ phi_n.pdf(X) for phi_n in self.estimates ]
		return ( np.hstack( [ -P_i * np.exp( P_i ) for P_i in P ] ).sum(1) )

	def contourPlot(self, S, fig, xrange, yrange, xstep, ystep, title="derived contour plot",axes=(0,1) ):
		(N,d) = S.shape
		xN = int((xrange[1]-xrange[0])/xstep)
		yN =  int((yrange[1]-yrange[0])/ystep)
		X = dstack(mgrid[xrange[0]:xrange[1]:xstep,yrange[0]:yrange[1]:ystep]).reshape([ xN *yN,2])
		x = arange(xrange[0],xrange[1],xstep)
		y = arange(yrange[0],yrange[1],ystep)

		fig.contourf(x,y,self.pdf(X).reshape([xN,yN]).T,200, antialiased=True, cmap=cm.gray )
		CS = plt.contour(x,y,self.pdf(X).reshape([xN,yN]).T, [.01,], colors='r' )
		fig.plot( hsplit( S,d )[ axes[0] ],hsplit( S,d )[ axes[1] ], 'r+' )
		#fig.clabel(CS, inline=1, fontsize=10)
		fig.axis( [ xrange[0],xrange[1],yrange[0],yrange[1] ] )
		fig.set_title(title)
		
