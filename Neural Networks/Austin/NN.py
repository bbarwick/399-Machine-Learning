
import numpy
import time
from numpy import exp, array, random, dot


class Network(object):

	step = 3.0
	
	def __init__(self, sizes):
		self.m = len(sizes)
		self.sizes = sizes
		self.b = [random.randn(y, 1) for y in sizes[1:]]
		self.w = [random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

	def feedforward(self,X):
		#begin trainning set by getting the calculated output
		for b,w in zip(self.b, self.w):
			X = self.sig(dot(w,X)+b)
		return X 
	def sig(self,z):
		return 1.0/(1.0+exp(-z))

	def sig_prime(self,z):
		return self.sig(z)*(1-self.sig(z))

	def feedforward(self,X):
		#begin trainning set by getting the calculated output
		for b,w in zip(self.b, self.w):
			X = self.sig(dot(w,X)+b)
		return X
	
	def partial_derivatives(self, X, a):   #X is input, a is desired output
		#Db = partials with respect to bias
		#Dw = partials with respect to weights

		Db= [numpy.zeros(b.shape) for b in self.b]
		Dw = [numpy.zeros(w.shape) for w in self.w]
	
		#calculated output:
		activation = X
		activations = [X] # list to store all the activations, layer by layer
		zs = [] # list to store all the z vectors, layer by layer
		for b, w in zip(self.b, self.w):
			z = dot(w, activation)+b
			zs.append(z)
			activation = self.sig(z)
			activations.append(activation)

		#first we have to find the last layer's partials
		#using Cost function 1/2n||y(x) - a||^2
		#dC/zL = dC/aL*sig'(zL)

		#partial of the Cost function with respect to calculated output:
		da = a - activations[-1]*(1-activations[-1])
		
		#dC/zL = DzL  partial of the last layer
		DzL = da*self.sig_prime(zs[-1])
		
		Db[-1] = DzL
		
		Dw[-1] = dot(DzL,activations[-2].transpose())	
		
		#preparing the recursive acumulation of the partial derivatives throught the layers
		Dzl = DzL

		#Dzjl = sum(Dzk(l+1)*wkj(l+1)*sig_prime(zjl)

		for i in xrange(self.m-2):
			l = self.m - 3 - i  #at i = 0, i is the second to last layer
			
			sig_p = self.sig_prime(zs[l])
	
			Dzl = dot(self.w[l+1].transpose(), Dzl)
			Dzl = sig_p*Dzl
			Db[l] = Dzl

			#print activations[l].shape
			#print Dzl.shape
			Dw[l] = (dot(Dzl,activations[l].transpose()))

		#add ajustments to the layers
 		for x in xrange(len(self.w)):
 			self.w[x] += self.step*Dw[x]
 			self.b[x] += self.step*Db[x]
		return
		
	def train_SGD(self, training_data, epocks, mini_batch_size, step_size, test_data=None):
		#train using stochastic gradient descent.

		#number of training_data and test data
		if test_data: n_test = len(test_data)
		n = len(training_data)
		self.step = step_size/mini_batch_size

		for i in xrange(epocks):
			#by shuffling the training data, we get a random set of sets from the first 
			#mini_batch_size elements
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				for x, y in mini_batch:
					self.partial_derivatives(x,y)
			if test_data:
				print "testing data..."
				print "Epoch {0}: {1}/{2}".format(i, self.evaluate(test_data), n_test)
			else:
				print "Epoch {0} complete".format(i)

		return  
		
	def evaluate(self, test_data):
		#taken from the text book
		test_results = [(numpy.argmax(self.feedforward(x)), y)
		                for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)

 

import mnist_loader
print "loading..."
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()
print "loaded"
                        
net = Network([784, 30, 10])
net.train_SGD(training_data, 30, 10, 3.0, test_data=test_data)
print "finished"
	
	

