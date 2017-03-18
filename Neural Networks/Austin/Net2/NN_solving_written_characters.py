
# coding: utf-8

# In[5]:

import numpy
from numpy import exp, array, random, dot


# In[14]:

class layer(object):
    input = None
    w = None
    output = None
    m = None
    b = None
    weighted_input = None
    
    
    def __init__(self, input_size, output_size):
        self.m = int(input_size)
        self.n = int(output_size)
        self.input = numpy.zeros(shape=(self.m))
        self.weighted_input = numpy.zeros(shape=(self.m))
        self.w = numpy.zeros(shape=(self.n,self.m))
        self.b = numpy.zeros(shape=(self.n))
        self.output = numpy.zeros(shape=(self.n))
        #initializing all weights in this layer to a random number 
        for x in xrange(self.m):
            for y in xrange(self.n):
                 self.w[y][x] = random.random(1)
            self.input[x] = 1
            self.b[y] = random.random(1)
        return 
    
#    
#    def partial_of(self,x,y):
#            #since there is only one route to anyone weight in a single layer
#            #it is just a single calculation
#            if self.last_layer:
#                return self.input[x] #because it doesn't have a sigmoid
#            else:
#                return self.output[y]*(1-self.output[y])*self.input[x]
        
#    def partial_through(self):
#        partial = numpy.zeros(shape=(self.m))
#        #getting all the partial derivatives of this layer, to send to the next
#        for x in xrange(self.m):
#            #from neuron n, therefore d(z[n])/d:
#            if self.last_layer:
#                partial_output = 1  #because it doesn't have a sig, partial through is just the weight
#            else:
#                partial_output = self.output[x]*(1-self.output[x])
#            for y in xrange(self.m):
#                #which weight/input in this layer to take the partial through
#                #weight[y][x] = partial of product(x*w)/partial of input
#                partial[y] += self.w[y][x]*partial_output
#                #partial of x sigmoid through (y,x) weight
#                #print "Partial of %d sigmoid through (%d,%d) weight: %f" % (x,y,x, partial[y])            
#        return partial
        
    def get_output(self, Input):
        self.input = Input
        for x in xrange(self.n):
            temp = 0
            for y in xrange(self.m):
                #getting all inputs through the wxy weight for input y   w(neuron,input)
                temp += self.input[y]*self.w[x][y] + self.b[x]
            #running the sum the product of weights*inputs through the sigmoid
            self.weighted_input[x] = temp
            self.output[x] = self.sig(temp)
        return self.output
        
    def sig(self,a):
        return 1/(1 + exp(-a))       


# In[163]:

class network(object):
    layers = []
    m = 0
    step = 1
    gradent = numpy.zeros(shape = (m,m,m))
    step_size = 10
    
    def __init__(self, sizes):
        #creates m layers with m neurons in each layer
        self.m = len(sizes)
        self.sizes = sizes
        sizes_prime = numpy.zeros(shape = (len(sizes)+1))
        for x in xrange(len(sizes)):
            sizes_prime[x+1] = sizes[x]
        sizes_prime[0] = sizes[0]
        self.layers = [layer(sizes_prime[x],sizes_prime[x+1]) for x in xrange(self.m)]
        return
    
    def sig(self,z):
        return 1.0/(1.0+exp(-z))

    def sig_prime(self,z):
        return self.sig(z)*(1-self.sig(z))
    
    def run(self,X):
        #begin trainning set by getting the calculated output
        input = X
        for x in xrange(self.m):
            input = self.layers[x].get_output(input)
        #generated ouput of running the input through just once
        output = input
        return output
        #self.partial_derivatives(X, a, output)
    
    def train_SGD(self, training_data, epocks, mini_batch_size, step_size, test_data=None):
        #train using stochastic gradient descent.
        
        #number of training_data and test data
        if test_data: n_test = len(test_data)
        n = len(training_data)
        self.step = step_size
        
        for i in xrange(epocks):
            #by shuffling the training data, we get a random set of sets from the first 
            #mini_batch_size elements
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            print len(mini_batches)
            for mini_batch in mini_batches:
                for x, y in mini_batch:
                    self.partial_derivatives(x,self.run(x),y)
                if test_data:
                    print "Epoch {0}: {1}/{2}".format(i, self.evaluate(test_data), n_test)
                else:
                    print "Epoch {0} complete".format(i)

        return   
    
    def evaluate(self, test_data):
    #taken from the text book
        test_results = [(numpy.argmax(self.run(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def partial_derivatives(self, X, a, output):   #X is input, a is desired output
        #Db = partials with respect to bias
        #Dw = partials with respect to weights
        Db = [[0 for x in range(self.sizes[y])] for y in range(self.m)]
        Dw =  [[[0 for x in range(self.sizes[y])] for x in range(self.sizes[y])] for y in range(self.m)]

        
        #first we have to find the last layer's partials
        #using Cost function 1/2n||y(x) - a||^2
        #dC/zL = dC/aL*sig'(zL)
        
        #partial of the Cost function with respect to calculated output:
        da = [(a[x] - output[x]) for x in xrange(self.sizes[self.m-1])]
        
        #dC/zL = DzL  partial of the last layer
        DzL = [da[x]*self.sig_prime(self.layers[self.m-1].weighted_input[x]) for x in xrange(self.sizes[self.m-1])]
        Db[self.m-1] = DzL
        input = numpy.zeros(shape = (len(self.layers[self.m-1].input),1))
        for x in xrange(len(input)):
            input[x] = [self.layers[self.m-1].input[x]]
        Dw[self.m-1] = dot(DzL,input.transpose())
        
        #preparing the recursive acumulation of the partial derivatives throught the layers
        Dzl = DzL
        
        #Dzjl = sum(Dzk(l+1)*wkj(l+1)*sig_prime(zjl)
        
        for i in xrange(self.m-1):
            l = self.m - 2 - i  #at i = 0, i is the second to last layer
            sig_p = self.sig_prime(self.layers[l].weighted_input)  #sig_prime(zl)
            
            Dzl = dot(self.layers[l+1].w.transpose(), Dzl) 
            for j in xrange(len(Dzl)):
                Dzl[j] * sig_p
            Db[l] = Dzl
            
            input = numpy.zeros(shape = (len(self.layers[l].input),1))
            for x in xrange(len(input)):
                input[x] = [self.layers[l].input[x]]
            Dw[l] = dot(Dzl,input.transpose())
            
            
        #add ajustments to the layers
        for x in xrange(self.m):
            for i in xrange(len(self.layers[x].w)):
                for j in xrange(len(self.layers[x].w[i])):
                    self.layers[x].w[i][j] += self.step*Dw[x][i][j]
            for i in xrange(len(self.layers[x].b)):
                self.layers[x].b[i] += self.step*Db[x][i]
        return

