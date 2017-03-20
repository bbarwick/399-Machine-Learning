import numpy
from numpy import exp, array, random, dot
import abc

from Activation import *

class Cost(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def f(a,y):
		return
		
    @abc.abstractmethod
    def df(a,y,z):
    	return
    	
class quadratic_cost(Cost):
	def __init__(self,act):
		self.activation = act

	def f(self,a,y):
		return 0.5*numpy.linalg.norm(a-y)**2
		
	def df(self,a,y,z):
		return (a-y) * self.activation.df(z)

class cross_enthropy_cost(Cost):
	def __init__(self,act):
		self.activation = act
		
	def f(self,a,y):
		return numpy.sum(numpy.nan_to_num(-y*numpy.log(a)-(1-y)*numpy.log(1-a)))
		
	def df(self,a,y,z):
		return (a-y)
		
#class log_likelyhood(Cost):
#	@staticmethod
#	def f(a,y):
#		return -log(a)
#	
#	@staticmethod
#	def df(a,y,z):
#		return
