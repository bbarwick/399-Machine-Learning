import numpy
from numpy import exp, array, random, dot

import abc

class Cost(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def f(a,y):
		return
		
    @abc.abstractmethod
    def df(a,y,z):
    	return
    	
    @abc.abstractmethod
    def act(z):
    	return
    
    @abc.abstractmethod
    def dact(z):
    	return
    	
class quadratic_cost(Cost):
	@staticmethod
	def f(a,y):
		return 0.5*numpy.linalg.norm(a-y)**2
		
	@staticmethod
	def df(a,y,z):
		return (a-y) * quadratic_cost.dact(z)
		
	@staticmethod
	def act(z):
		return 1.0/(1.0+exp(-z))
	
	@staticmethod
	def dact(z):
		return quadratic_cost.act(z)*(1-quadratic_cost.act(z))

class cross_enthropy_cost(Cost):
	@staticmethod
	def f(a,y):
		return numpy.sum(numpy.nan_to_num(-y*numpy.log(a)-(1-y)*numpy.log(1-a)))
		
	@staticmethod
	def df(a,y,z):
		return (a-y)
		
	@staticmethod
	def act(z):
		return 1.0/(1.0+exp(-z))
	
	@staticmethod
	def dact(z):
		return cross_enthropy_cost.act(z)*(1-cross_enthropy_cost.act(z))
