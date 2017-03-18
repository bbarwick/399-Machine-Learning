import numpy
import time
from numpy import exp, array, random, dot

from Cost import quadratic_cost, cross_enthropy_cost

class Network(object):

	step = 3.0
	
	def __init__(self, sizes):
		self.m = len(sizes)
		self.sizes = sizes
		self.b = [random.randn(y, 1) for y in sizes[1:]]
		self.w = [random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
		
	def sig(self,z):
		return 1.0/(1.0+exp(-z))

	def sig_prime(self,z):
		return self.sig(z)*(1-self.sig(z))

	def feedforward(self,X):
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
		#da = cost_function(1,a,activations[-1])
		#da = (a - activations[-1])#/self.sig_prime(zs[-1])
		
		#dC/zL = DzL  partial of the last layer
		DzL = self.Cost.df(a,activations[-1],zs[-1])				#da*self.sig_prime(zs[-1])
		
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

			Dw[l] = (dot(Dzl,activations[l].transpose()))

		#add ajustments to the layers
 		for x in xrange(len(self.w)):
 			self.w[x] += self.step*Dw[x]
 			self.b[x] += self.step*Db[x]
		return
		
	def train_SGD(self, training_data, epocks, mini_batch_size, step_size, test_data=None, cost = 0):
		#train using stochastic gradient descent.
		
		#cost function selected
		if(cost == 1):
			self.Cost = cross_enthropy_cost
		else:
			self.Cost = quadratic_cost
		
		
		accy = 0
		
		#number of training_data and test data
		if test_data: n_test = len(test_data)
		n = len(training_data)
		self.step = step_size/mini_batch_size

		for i in xrange(epocks):
			st= time.time()
			#by shuffling the training data, we get a random set of sets from the first 
			#mini_batch_size elements
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				for x, y in mini_batch:
					self.partial_derivatives(x,y)
			if test_data:
				print "testing data..."
				eva = self.evaluate(test_data)
				if (eva > accy):
					self.save_weights("auto_save.txt")
					accy = eva
				print "Epoch {0}: {1}/{2}".format(i, eva, n_test)
			else:
				print "Epoch {0} complete".format(i)
		return  
		
	def evaluate(self, test_data):
		#taken from the text book
		test_results = [(numpy.argmax(self.feedforward(x)), y)
		                for (x, y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)
	
	def save_weights(self, file_name):
		save = open(file_name, 'w')
		save.write(str(self.sizes[0]))
		for x in xrange(len(self.sizes)-1):
			save.write(",")
			save.write(str(self.sizes[x+1]))
		save.write("\n")
		
		for x in xrange(self.m-1):
			save.write(" ")
			for y in xrange(len(self.w[x])):
				for z in xrange(len(self.w[x][y])):
					save.write(str(self.w[x][y][z]))
					save.write(",")
				save.write("|")
		save.write("\n")
		
		for x in xrange(self.m-1):
			save.write(" ")
			for y in xrange(len(self.b[x])):
				save.write(str(self.b[x][y][0]))
				save.write(",")
		save.close()
		return
		
	def load_weights(self, file_name):
		load = open(file_name, 'r')
		File = load.read()
		data = File.split("\n")
		init = data[0]
		weights = data[1].split(" ")
		bias = data[2].split(" ")
		weights.remove('')
		bias.remove('')
		
		#init
		
		#load weights:
		for x in xrange(self.m-1):
			w = weights[x].split("|")
			for y in xrange(len(self.w[x])):
				w1 = w[y].split(",")
				for z in xrange(len(self.w[x][y])):
					self.w[x][y][z] = float(w1[z])
	
		#load bias:
		for x in xrange(self.m-1):
			b = bias[x].split(",")
			for y in xrange(len(self.b[x])):
				self.b[x][y][0] = b[y]
		return

 

import mnist_loader
print "loading..."
training_data, validation_data, test_data = \
mnist_loader.load_data_wrapper()
print "loaded"
                    
net = Network([784, 30, 10])
#net.load_weights("auto_save.txt")
net.train_SGD(training_data, 30, 10, 3.0, test_data=test_data,cost=0)
#net.save_weights("characters2.txt")

net.load_weights("auto_save.txt")
print "Max trainning:"
print "Epoch 0: {0}/{1}".format(net.evaluate(test_data), len(test_data))
print "finished"
	
	
