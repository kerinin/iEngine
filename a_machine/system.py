#! /usr/bin/env python
from math import *
import numpy as np

class model:
  def __init__(self):
    # create first layer
    pass
    
  def process(point):
    # pass point to first layer
    pass
      
class layer:
  def __init__(self, lower_level=None):
    # set lower level
    self.lower_level = lower_level
    self.upper_level = None
    self.sequences = []
    
    # set lower level's upper level
    if lower_level:
      lower_level.upper_level = self
    
    self.vectors = []

  def process(point):
    # append sequence to vectors
    self.sequences += (self.sequences[-1][1:] + [point])
    
    # calculate vector activation
    self.activation = activation_function(self.sequences, self.sequences[-1])

    # pass activation vector to next layer (possibly create next layer)
    if self.sequences.count == 1:
      layer(self)
    
    self.upper_level.process( self.sequences.map(activation) )

		
