import cv2
import numpy
from matplotlib import pyplot as plt

import NN


net = NN.Network([784, 30, 10])
net.load_weights("characters.txt")

base_file = '/home/austin/Documents/Machine Learning Ind/python/practice/opencv practice/'

numbers = ['zero.jpg','one.jpg','two.jpg','three.jpg', 'eight.jpg', 'five.jpg', 'nine.jpg']

for i in xrange(len(numbers)):

	img = cv2.imread(base_file + numbers[i])

	for x in xrange(len(img)):
		for y in xrange(len(img[x])):
		    temp = img[x][y][2]
		    img[x][y][2] = img[x][y][0]
		    img[x][y][0] = temp

	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	new_image = numpy.zeros(shape = (784,1))
	for x in xrange(len(gray_image)):
		for y in xrange(len(gray_image[x])):
		    new_image[28*x + y][0] = 1 - gray_image[x][y]/255.0
		    


	result = net.feedforward(new_image)
	print "Testing: %s, result: %d" % (numbers[i],numpy.argmax(result))

