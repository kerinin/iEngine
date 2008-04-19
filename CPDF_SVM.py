

class SVM:
# A CPDF estimator based on the SVM architecture.
# 
# takes one parameter which controls how many support vectors from the data set
# are used in the estimation.  This parameter controls both the compression
# generalization, and the smoothing of the resulting function
	
	gamma = None						# smoothness / sparseness parameter for kernel computations
	
	def __init__(self, gamma=None):
		self.gamma = gamma
		
	def optimize(self, data, gamma):
	# determines the optimal SV's and weights for the given data with the given smoothing parameter
		
		# SVM stuff here...
		
		return CPDF(SV,beta)
		
class CPDF:
# A container for the optimiztion parameters which provides regression and PDF's based on 
# those parameters
	SV = list()
	beta = list()
	
	def __init__(self,SV,beta):
		self.SV = SV
		self.beta = beta
		
	def reg(self,x):
		pass
		
	def den(self,x):
		pass