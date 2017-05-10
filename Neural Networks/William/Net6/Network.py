import numpy
import time
from numpy import exp, array, random, dot

from Cost import *
from Activation import *

class Network(object):
	
	alt_out = False
	
	def __init__(self, sizes):
		self.m = len(sizes)
		self.sizes = sizes
		self.b = [2*random.randn(y, 1) for y in sizes[1:]]
		self.w = [2*random.rand(y, x) - 1 for x, y in zip(sizes[:-1], sizes[1:])]
		
		#set default activation and cost function
		self.activation = sigmoid
		self.alt_act = sigmoid
		self.cost = quadratic_cost(sigmoid)
		
	def setActivation(self, act):
		self.activation = act
		return
	
	def setCost(self, cost):
		self.cost = cost(self.activation)
		return
		
	def alt_output(self,act):
		self.alt_out = True
		self.alt_act = act
		return
	
	def feedforward(self,X):
		activation = X
		activations = [X] # list to store all the activations, layer by layer
		zs = [] # list to store all the z vectors, layer by layer
		#print X
		#return dum
		for b, w in zip(self.b, self.w):
			z = dot(w, activation)+b
			zs.append(z)
			activation = self.activation.f(z)
			activations.append(activation)
			#print "BBBBBBBBBBBBBbbbbbb"
			#print b
			#print "WWWWWWWWWWWWWWWWWwww"
			#print w
			#print "zzzzzzzzzzzzzzzzzzzzzZ"
			#print z
			#print "ACACACACACACT"
			#print activation

		#return dum
		#alternate output layer:
		if self.alt_out:
			activations[-1] = self.alt_act.f(zs[-1])
		return activations[-1]
	
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
		#	print "WWWWWWWWWWWWWWWWWW"
		#	print dot(w,activation)
		#	print "BBBBBBBBBBBBBBBBBBBBBBb"
		#	print b
		#	print "ZZZZZZZZZZZZZZZZZZZZZz"
		#	print z
			zs.append(z)
			activation = self.activation.f(z)
			activations.append(activation)

		#alternate output layer:
		if self.alt_out:
			activations[-1] = self.alt_act.f(zs[-1])
			
		#return
		#first we have to find the last layer's partials
		#using Cost function 1/2n||y(x) - a||^2
		#dC/zL = dC/aL*sig'(zL)

		#partial of the Cost function with respect to calculated output:
		#da = cost_function(1,a,activations[-1])
		#da = (a - activations[-1])#/self.sig_prime(zs[-1])
		
		#dC/zL = DzL  partial of the last layer
		DzL = -1*self.cost.df(a,activations[-1],zs[-1])				#da*self.sig_prime(zs[-1])
		
		Db[-1] = DzL
		
		Dw[-1] = dot(DzL,activations[-2].transpose())	
		
		#preparing the recursive acumulation of the partial derivatives throught the layers
		Dzl = DzL

		#Dzjl = sum(Dzk(l+1)*wkj(l+1)*sig_prime(zjl)

		for i in xrange(self.m-2):
			l = self.m - 3 - i  #at i = 0, i is the second to last layer
			
			act_p = self.activation.df(zs[l])
			
			#print dot(self.w[l+1].transpose(), Dzl)
			
			Dzl = dot(self.w[l+1].transpose(), Dzl)
			Dzl = act_p*Dzl
			Db[l] = Dzl

			Dw[l] = (dot(Dzl,activations[l].transpose()))
			
		return (Dw, Db)
		
	def train_SGD(self, training_data, epocks, mini_batch_size, step_size, lmbda, test_data=None):
		#train using stochastic gradient descent.	
		
		accy = 0
		
		#number of training_data and test data
		if test_data: n_test = len(test_data)
		n = len(training_data)
		self.n = n
		self.step = step_size
		self.mini_batch_size = mini_batch_size
		self.lamb = lmbda

		for i in xrange(epocks):
			st= time.time()
			#by shuffling the training data, we get a random set of sets from the first 
			#mini_batch_size elements
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
			for mini_batch in mini_batches:
				Dw_sum = [numpy.zeros(w.shape) for w in self.w]
				Db_sum = [numpy.zeros(b.shape) for b in self.b]
				for x, y in mini_batch:
					Dw, Db = self.partial_derivatives(x,y)
					Dw_sum = [dw_s + dw for dw_s, dw in zip(Dw_sum, Dw)]
					Db_sum = [db_s + db for db_s, db in zip(Db_sum, Db)]
				#print self.w[0][0]
				#print (step_size/mini_batch_size)
				#print ((step_size/mini_batch_size)*Dw_sum[0])[0]
				self.w = [(1-step_size*(lmbda/n))*w - (step_size/mini_batch_size)*nw for w, nw in zip(self.w, Dw_sum)]
				self.b = [b-(step_size/mini_batch_size)*nb for b, nb in zip(self.b, Db_sum)]			
				#return
	
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
		print self.feedforward(test_data[0][0])
		print test_data[0][1]
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



#import mnist_loader
#print "loading..."
#training_data, validation_data, test_data = \
#mnist_loader.load_data_wrapper()
#print "loaded"

#net = Network([784, 30, 10])
#net.setActivation(sigmoid)
#net.setCost(log_likelyhood_cost)
#net.alt_output(softmax)
#net.load_weights("auto_save.txt")
#net.train_SGD(training_data, 30, 10, 0.5, 5.0, test_data=test_data)
#net.save_weights("characters2.txt")

#net.load_weights("auto_save.txt")
#print "Max trainning:"
#print "Epoch 0: {0}/{1}".format(net.evaluate(test_data), len(test_data))
#print "finished"

