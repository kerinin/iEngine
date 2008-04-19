
from system_2 import input_base, function_base

class input_svm(input_base):
	
	def optimize(self,t_delta,time=None):
		raise StandardError, 'This function not implemented'
		
	def estimate(self, time=None, hypotheses = None):
	# estimates the input's value at time under the constraints that at the time/value pairs
	# in hypothesis the input has the specified values.
		
		raise StandardError, 'This function not implemented'

class function_svm(function_base):
# a function describing the behavior of an input over a specific observed time interval
		
	def __sub__(self,a):
	# overloads the subtract function to compute distance between two functions or a function and a cluster
		raise StandardError 'This function not implemented'
		
	def optimize(self,data,CPDF,*args,**kargs):
	# optimizes data using the specified Conditional Probability Distribution Function estimator
		raise StandardError 'This function not implemented'
		
	def reg(self,t):
	# evaluates the most function at time t
		raise StandardError 'This function not implemented'
		
	def den(self,t):
	# returns a probability density over the range of the function at time t
		raise StandardError 'This function not implemented'
