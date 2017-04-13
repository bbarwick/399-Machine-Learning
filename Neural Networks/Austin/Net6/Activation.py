import numpy
from numpy import exp, array, random, dot

import abc

class Activation(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def f(z):
		return
		
    @abc.abstractmethod
    def df(z):
    	return
  	
class sigmoid(Activation):
	@staticmethod
	def f(z):
		return 1.0/(1.0+exp(-z))
	
	@staticmethod
	def df(z):
		return sigmoid.f(z)*(1-sigmoid.f(z))
		
class softmax(Activation):
	@staticmethod
	def f(z):
		return exp(z)/sum(exp(z))
		
	@staticmethod
	def df(z):
		return 1

