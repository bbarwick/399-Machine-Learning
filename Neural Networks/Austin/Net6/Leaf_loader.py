import cv2
import numpy as np
from matplotlib import pyplot as plt

class Leaf_loader(object):
	data = []
	def __init__(self):
		load_train = open("../../python/Leaf_classification_data/train.csv", 'r')
		train = load_train.read()
		self.leafs_train = train.split("\n")
		self.leafs_train.pop(0)
		self.leafs_train.pop(len(self.leafs_train) - 1)
		
		self.leaf_id = {}
		i = 0

		self.Leafs = []

		for leaf in self.leafs_train:
			single_leaf = leaf.split(",")
			if(not(single_leaf[1] in self.leaf_id)):
				self.leaf_id[single_leaf[1]] = i
				i = i+1
			ID = single_leaf[0]
			classification = self.leaf_id[single_leaf[1]]
			single_leaf.pop(0)
			single_leaf.pop(0)
			data = np.zeros(shape = (len(single_leaf),1))
			for x in xrange(len(single_leaf)):
				data[x,0] = float(single_leaf[x])
			self.Leafs.append((data,classification,ID))




	def save_picture_vectors(self):
		save = open("training.txt", 'w')
		for i in xrange(len(self.Leafs)):
			print i
			save.write(str(self.Leafs[i][1]))
			save.write(",")
			
			image = cv2.imread("../../python/Leaf_classification_data/images/" + str(self.Leafs[i][2]) + ".jpg")
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			#print image.shape			#for testing purposes. fx and fy are the multipliers for the height and width of the picture
			image = cv2.resize(image, (0,0), fx=0.15, fy=0.15)
			#print image.shape[0]*image.shape[1]
			#plt.imshow(image)
			#plt.show()
			#return
	
			image_vec = []
			for x in xrange(len(image)):
				for y in xrange(len(image[x])):
					z = image[x][y]
					if int(z) != 0:
						save.write(str(x*image.shape[0] + y) + "|" + str(z))
						save.write(",")
			save.write("\n")
		save.close()
	
	def load_picture(self):
		load = open("training.txt", 'r')
		self.leaf_input = []
		for i in xrange(990):
			print i
			i = i+1
			fil = load.readline()
			self.data.append(fil)
			leaf = np.zeros(shape = (38688,1))
			
			datum = fil.split(",")
			for x in xrange(len(datum)-2):
				one = datum[x+1].split("|")
				leaf[int(one[0])][0] = int(one[1])/255.0
			self.leaf_input.append((leaf,datum[0]));
		print len(self.data)
		load.close()


load = Leaf_loader()
print "Started:"

#load.save_picture_vectors()
load.load_picture()
#print (load.data[0].split(","))
print "Complete:"



#training_data = load.Leafs
#test_data = load.Leafs[(len(load.Leafs)*2)/33:]

#for x in xrange(len(load.Leafs)):
#	load.Leafs[x] = (load.Leafs[x][0], int(load.Leafs[x][1])/3)

#print load.Leafs[0][0]
#for leaf in load.Leafs:
#	print leaf[1]

from Network import *
from Cost import *
from Activation import *

#net = Network([192, 10, 3])
#net.setActivation(sigmoid)
#net.setCost(quadratic_cost)
#net.save_weights("random_weights.txt")
#net.load_weights("random_weights.txt")
#data epocks minibatch step lambda
#net.train_SGD(training_data, 30, 50, 3.0, 3.0, test_data=load.Leafs)
#net.save_weights("characters2.txt")

#net.load_weights("auto_save.txt")
#print "Max trainning:"
#print "Epoch 0: {0}/{1}".format(net.evaluate(load.Leafs), len(load.Leafs))
#print "finished"

