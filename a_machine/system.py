#! /usr/bin/env python
from math import *
import numpy as np

import cs_divergence

# NOTES:
# timestamping
# timeshifting
# sparse arrays (needed since the dimensionality of the input increases with each observation)
# normalize activation?
#   The divergence is based on the PDF.  If we use the full history, the divergence
#   will depend on the length of the full history, even if it's the same
#   in the 'neighborhood' of a given sequence
# we're assuming that points are processed in the order they occur time-wise.  
# we're assuming we can always pull previous points from previous sequences
#   this breaks for the first points


class model:
  def __init__(self):
    # create first layer
    self.layers = [ layer(self) ]
    self.born = datetime.now()
    
  def process(self, point, time=None):
    if not time:
      time = (datetime.now() - self.born).total_seconds()
      
    # pass point to first layer
    self.layers[0].process(point)
  
  def cycle(self, time=None):
    if not time:
      time = (datetime.now() - self.born).total_seconds()
      
    # calculate activation
    
    # descend hierarchy
    
    # integrate base activation
      
class layer:
  def __init__(self, model, sequence_length=2, lower_level=None):
    # set lower level
    self.model = model
    self.lower_level = lower_level
    self.upper_level = None
    self.sequence_length = sequence_length
    self.last_time = None
    
    # [sequence][point][dimension]
    self.sequences = np.array([]).reshape(0,0,0).astype('float32')

  def activation(self, time=0):
    # this should be taken from a reference set
    return cs_divergence.from_many( self.sequences, self.sequences[-1] )
  
  # point => [dimension]
  def process(self, point, time=None):
    if not time:
      time = (datetime.now() - self.model.born).total_seconds()

      
    point = nd.array( [0, point], [len(point)+1] ).astype('float32')
      
    # append point to sequences
    # pulls the last (dimension-1) points from the last sequence,
    # shifts the previous points backward so the sequence is zeroed at 'time'
    # appends 'point' the the end, and adds it to the sequences array
    points_from_previous = self.sequences[-1,1:]
    if self.last_time:
      points_from_previous[:,0] = points_from_previous[:,0] - time + self.last_time
    np.append(self.sequences, points_from_previous + [point], 0)
    
    # create upper level if it doesn't exist
    if not self.upper_level:
      self.upper_level = layer( self.sequence_length, self)
    
    # pass activation to upper level if 'sequence_length' points have been observed since last pass
    if not self.sequences.shape[0] % self.upper_level.sequence_length:
      # calculate vector activation
      self.upper_level.process( self.activation() )
      
    self.last_time = time

		
