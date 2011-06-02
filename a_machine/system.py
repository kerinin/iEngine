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
    self.layer_number = 0 if lower_level == None else lower_level.layer_number + 1
    
    # [sequence][point][dimension]
    self.sequences = None

  # time: time since last observation
  def activation(self, time=0):
    return cs_divergence.from_many( self.sequences, self.sequences[-1], self.gamma )
  
  # point => array()[dimension]
  def process(self, point, time=None, extra_dim=None):
    print "processing point on layer %s" % self.layer_number

    if not time:
      time = total_seconds(datetime.now() - self.model.born)
          
    # prefix the point with time value (0), format correctly
    point = np.insert(point,0,0).astype('float32')
      
    # behavior for new layers
    if self.sequences == None:
      
      # behavior for first run
      if self.point_cache == None:
        self.point_cache = np.array([point]).reshape(1,point.shape[0]).astype('float32')
        
      # behavior for points before sequence length reached
      else:
        # add extra dimensions
        if not extra_dim == None:
          print self.point_cache.shape
          self.point_cache = np.hstack((self.point_cache,extra_dim.reshape(1,extra_dim.shape[0])))
          print self.point_cache.shape
          
        # shift existing points along time axis
        self.point_cache[:,0] = self.point_cache[:,0] - time + self.last_time
        # add the new point
        self.point_cache = np.vstack((self.point_cache,point))
        
        if self.point_cache.shape[0] == self.sequence_length:
          # converte point cache into sequence list
          self.sequences = np.array(self.point_cache).reshape(1,self.point_cache.shape[0],self.point_cache.shape[1]).astype('float32')
          
          # cleanup
          self.point_cache = None
          
    # behavior for layers with at least one previous sequence
    else:
      # add extra dimensions
      if not extra_dim == None:
        self.point_cache = np.dstack((self.sequences,extra_dim.reshape(1,1,extra_dim.shape[0])))
        
      # pull the tail from the last sequence
      points_from_previous = self.sequences[-1][1:]
      
      # shift the points in time
      points_from_previous[:,0] = points_from_previous[:,0] - time + self.last_time
      
      # add the point to make a sequence and add it to the list
      sequence = np.vstack((points_from_previous, point))
      self.sequences = np.vstack((self.sequences, sequence.reshape(1,self.sequence_length,sequence.shape[1])))
        
      # create upper level if it doesn't exist
      if not self.upper_level:
        self.upper_level = layer( self.model, 2, self)
      
      # pass activation to upper level if 'sequence_length' points have been observed since last pass
      print "%s sequences" % self.sequences.shape[0]
      if not ((self.sequences.shape[0]) % self.upper_level.sequence_length):
        # calculate vector activation
        extra_dim = cs_divergence.from_many( self.sequences[:-1,:,:], self.sequences[-1], self.gamma )
        print self.sequences.shape
        print extra_dim.shape
        self.upper_level.process( 
          self.activation(),
          extra_dim = extra_dim
        )
    
    self.last_time = time