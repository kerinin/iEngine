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

from SVM_D2 import svm
	
	
class engine:
	def __init__(self, estimates, S):
		
		self.estimates = estimates
		
		self.Delta = np.vstack( [ self._H(s) for s in S ] )
			
		#self.varphi = svm( data=self.Delta, Lambda=.00005, gamma=[.125,.25,.5,1,2,4,8,16] )
		self.varphi = svm( data=self.Delta, Lambda=.0005, gamma=[16,32,64,128,256,512] )
		
	def pdf(self,X,S):
	# @param X		[Nxd] array of points for which to calculate a probability value
	# @param X		[Nxd] array of sample points for calculating entropy of existing estimates
		
		prior = np.hstack( [ phi_n.pdf(X) for phi_n in self.estimates ] )
		H_S = self._H( S )
		
		print self.varphi.pdf(self.Delta)
		print self.varphi.beta
		print self.varphi.beta.sum()
		
		return ( prior * ( self._varphi( H_S ) / self._varphi( H_S + self._H( X ) ) ) ).prod(1)
		
	def _varphi( self, delta ):
	# Probability distribution of known estimates' entropy

		return self.varphi.pdf( delta )
		
	def _H(self,S):
	# @param X		[Nxd] array of observations
		
		(N,d) = S.shape
		P = [ phi_n.pdf(S) for phi_n in self.estimates ]
		
		return ( np.hstack( [ -P_i * np.exp( P_i ) for P_i in P ] ).sum(0)/N ).reshape([1,len(self.estimates)])
	
	def varphiPlot( self, fig, axes=(0,1) ):
		fig.plot( np.hsplit(self.varphi.X,self.varphi.d)[axes[0]], np.hsplit(self.varphi.X,self.varphi.d)[axes[1]], 'ro' )
		
	def contourPlot(self, S, fig, xrange, yrange, xstep, ystep, title="derived contour plot",axes=(0,1) ):
		(N,d) = S.shape
		xN = int((xrange[1]-xrange[0])/xstep)
		yN =  int((yrange[1]-yrange[0])/ystep)
		X = np.dstack(np.mgrid[xrange[0]:xrange[1]:xstep,yrange[0]:yrange[1]:ystep]).reshape([ xN *yN,2])
		x = np.arange(xrange[0],xrange[1],xstep)
		y = np.arange(yrange[0],yrange[1],ystep)

		fig.contourf(x,y,self.pdf(X,S).reshape([xN,yN]).T,200, antialiased=True, cmap=cm.gray )
		CS = plt.contour(x,y,self.pdf(X,S).reshape([xN,yN]).T, [.01,], colors='r' )
		fig.plot( np.hsplit( S,d )[ axes[0] ],np.hsplit( S,d )[ axes[1] ], 'r+' )
		#fig.clabel(CS, inline=1, fontsize=10)
		fig.axis( [ xrange[0],xrange[1],yrange[0],yrange[1] ] )
		fig.set_title(title)
		
