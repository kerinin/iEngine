#! /usr/bin/env python
from math import *
from datetime import *
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

def total_seconds(td):
  return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / 10**6

class model:
  def __init__(self, born=None):
    if not born:
      born = datetime.now()
      
    # create first layer
    self.layers = [ layer(self) ]
    self.born = born
    
  def process(self, point, time=None):
    if not time:
      time = total_seconds(datetime.now() - self.born)
      
    # pass point to first layer
    self.layers[0].process(point)
  
  def cycle(self, time=None):
    if not time:
      time = total_seconds(datetime.now() - self.born)
      
    # calculate activation
    
    # descend hierarchy
    
    # integrate base activation
      
class layer:
  def __init__(self, model, sequence_length=2, lower_level=None, gamma=.1):
    # set lower level
    self.model = model
    self.lower_level = lower_level
    self.upper_level = None
    self.sequence_length = sequence_length
    self.last_time = None
    self.point_cache = None
    self.gamma = gamma
    
    # [sequence][point][dimension]
    self.sequences = None

  # time: time since last observation
  def activation(self, time=0):
    return cs_divergence.from_many( self.sequences, self.sequences[-1], self.gamma )
  
  # point => array()[dimension]
  def process(self, point, time=None, extra_dim=None):
    point = np.insert(point,0,0).astype('float32')
    
    if not time:
      time = total_seconds(datetime.now() - self.model.born)

    if self.point_cache == None:
      self.point_cache = np.array([point]).reshape(1,point.shape[0]).astype('float32')
      
    else:
      # append point to sequences
      # pulls the last (dimension-1) points from the last sequence,
      # shifts the previous points backward so the sequence is zeroed at 'time'
      # appends 'point' the the end, and adds it to the sequences array
      points_from_previous = self.point_cache[1:] if self.point_cache.shape[0] == self.sequence_length else self.point_cache
      if self.last_time:
        points_from_previous[:,0] = points_from_previous[:,0] - time + self.last_time
        
      sequence = np.vstack((points_from_previous, point))
    
      if sequence.shape[0] == self.sequence_length:
      
        if extra_dim:
          pass
        
        if self.sequences == None:
          self.sequences = np.array([sequence]).reshape(1,self.sequence_length,sequence.shape[1]).astype('float32')
        else:
          self.sequences = np.vstack((self.sequences, sequence.reshape(1,self.sequence_length,sequence.shape[1])))
          
      
        # create upper level if it doesn't exist
        if not self.upper_level:
          self.upper_level = layer( self.model, self.sequence_length, self)
        
        # NOTE: this is going to require figuring out that dimensional expansion issue
        # pass activation to upper level if 'sequence_length' points have been observed since last pass
        #if not (self.sequences.shape[0] % self.upper_level.sequence_length):
          # calculate vector activation
          #self.upper_level.process( self.activation() )
              
      self.point_cache = sequence
    

    self.last_time = time