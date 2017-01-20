
# coding: utf-8

#William Clayton

# In[15]:

import numpy
from numpy import exp, array, random, dot


# In[254]:

class layer(object):
    input = None
    w = None
    output = None
    m = None
    last_layer = False
    def __init__(self, m, last_layer):
        self.input = numpy.zeros(shape=(m))
        self.w = numpy.zeros(shape=(m,m))
        self.output = numpy.zeros(shape=(m))
        self.m = m
        self.last_layer = last_layer
        #initializing all weights in this layer to a random number 
        for x in xrange(m):
            for y in xrange(m):
                self.w[x][y] = 2*random.random(1)
            self.input[x] = 1
        return 
    
    def partial_of(self,x,y):
            #since there is only one route to anyone weight in a single layer
            #it is just a single calculation
            if self.last_layer:
                return self.input[x] #because it doesn't have a sigmoid
            else:
                return self.output[y]*(1-self.output[y])*self.input[x]
        
    def partial_through(self):
        partial = numpy.zeros(shape=(self.m))
        #getting all the partial derivatives of this layer, to send to the next
        for x in xrange(self.m):
            #from sig x, therefore dzx/d:
            if self.last_layer:
                partial_output = 1  #because it doesn't have a sig, partial through is just the weight
            else:
                partial_output = self.output[x]*(1-self.output[x])
            for y in xrange(self.m):
                #which weight/input in this layer to take the partial through
                #weight[y][x] = partial of product(x*w)/partial of input
                partial[y] += self.w[y][x]*partial_output
                #partial of x sigmoid through (y,x) weight
                #print "Partial of %d sigmoid through (%d,%d) weight: %f" % (x,y,x, partial[y])            
        return partial
        
    def get_output(self, Input):
        self.input = Input
        for x in xrange(self.m):
            temp = 0
            for y in xrange(self.m):
                #getting all inputs through the wyx weight for input y
                temp += self.input[y]*self.w[y][x]
            #running the sum the product of weights*inputs through the sigmoid
            if self.last_layer:
                self.output[x] = temp
            else:
                self.output[x] = self.sig(temp)
        return self.output
        
    def sig(self,a):
        return 1/(1 + exp(-a))
    


# In[285]:

class network(object):
    layers = []
    m = 0
    gradent = numpy.zeros(shape = (m,m,m))
    
    def __init__(self, m):
        #creates m layers with m neurons in each layer
        self.layers = [layer(m,False) for _ in xrange(m+1)]
        self.layers[m].last_layer = True
        self.m = m
        return
    
    def train_set(self, x, z):
        #begin trainning set by getting the calculated output
        input = x
        for x in xrange(self.m+1):
            input = self.layers[x].get_output(input)
        #generated ouput of running the input through just once
        output = input
        
        #now to calculate the gradient:
        #note at this time each layer's output variable is still equal to the output of that
        #layer when the input was sent through
        
        #dP/dz for each neuron:
        dz = numpy.zeros(shape = (self.m))
        for x in xrange(self.m):
            dz[x] = z[x] - output[x]
        print dz
        
        #gradent for each weight in each layer
        #(x,y,z) x: layer, (y,z): position of weight
        gradent = numpy.zeros(shape = (self.m,self.m,self.m))
        #getting the last layer since it doesn't have to go through any other layers
        for y in xrange(self.m):
            for z in  xrange(self.m):
                gradent[self.m-1][y][z] = self.layers[self.m-1].partial_of(y,z)
                #print "getting layer %d weight (%d,%d)" % (self.m-1,y,z)
                
        partial_through = numpy.zeros(shape = (self.m)) #culminating partials through the layers
        for x in xrange(self.m):
            partial_through[x] = 1    #setting the intial to 1 to keep a product tally
        
        for x in xrange(self.m-1):
            #starting from the second to last row, which is why its -2
            index = self.m - x - 2
            #variable to store each partial through the layers
            temp = numpy.zeros(shape = (self.m))
            temp = self.layers[index + 1].partial_through()
            for y in xrange(self.m):
                partial_through[y] *= temp[y]  #adding the previous layers partials to the running tally
            
            #compiling the current layer's partials with the previous layers
            for y in xrange(self.m):
                for z in xrange(self.m):
                    gradent[index][y][z] = self.layers[index].partial_of(x,y)*partial_through[y]
                    #print "getting layer %d weight (%d,%d)" % (index,y,z)
                 
        #adding the gradent to the respective weight
        for x in xrange(self.m):
            for y in xrange(self.m):
                for z in xrange(self.m):
                    self.layers[x].w[y][z] -= gradent[x][y][z]
        self.gradent = gradent
        
        return input


# In[286]:

l1 = layer(3, False)
net = network(3)
print l1.get_output([1,2,3])
print l1.partial_through()


# In[287]:

for x in xrange(1000):
    net.train_set([1,2,3],[1,2,3])
print net.train_set([1,2,3],[1,2,3])

#print net.gradent
#print net.layers[2].input


# In[ ]:



