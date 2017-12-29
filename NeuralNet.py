
"""
Authored by: Hager Radi
11 Oct. 2016
A basic implementation of a neural network model, implemeted with back propagtion and the training procedure; works for a dynamic number of layers
"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import cv2
from PIL import Image


random.seed(0)



training_data = []
training_labels = []

error_array = []
epoch_array = []

c1 = 0
c2 = 0

outputLabels = []

def ReadFromFile(filename):
	with open(filename) as f:
		for line in f:
			data = line.split()
			#print data
			if len(data)==0:
				return;
			training_data.append([float(data[0]),float(data[1])])
			training_labels.append(int(data[2]))

def Sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def Relu(x):
    return np.maximum(0,x)

def dRelu(x):
    return (x>0)

def TAN(x,deriv=False):
    if(deriv==True):
        return 1.0 - x**2
    return math.tanh(x)

def rand(a, b):
    return (b-a)*random.random() + a

def createMatrix(M, N, fill = 0.0):
    m = []
    for i in range(M):
        m.append([fill]*N)
    return m


class NeuralNet:
	def __init__(self, n_in , hidl , hid_n, n_out):
		self.n_in = n_in + 1
		self.n_out = n_out
		self.hidl = hidl + 2
		self.hid_n = hid_n

		self.Weights = [None] * (self.hidl)
		self.deltas = [None] * (self.hidl)
		self.ChangeM = [None] * (self.hidl)

		self.activations = [None] * (self.hidl)

		self.hid_n = [self.n_in] + self.hid_n + [self.n_out]

		for i in range(len(self.Weights)-1):
			self.Weights[i] = createMatrix(self.hid_n[i], self.hid_n[i+1], 1.0)
			self.deltas[i] = createMatrix(self.hid_n[i], self.hid_n[i+1], 1.0)
			self.ChangeM[i] = createMatrix(self.hid_n[i], self.hid_n[i+1])


		for i in range(len(self.hid_n)):
			self.activations[i] = [1.0]* self.hid_n[i]

		for i in range(len(self.Weights)-1):
			for j in range(len(self.Weights[i])):
				for k in range(len(self.Weights[i][j])):
					self.Weights[i][j][k] = 2*rand(-2.0, 2.0)-1
					#self.Weights[i][j][k] = 2*rand(len(self.Weights[i]), len(self.Weights[i][j]))-1

	def ForwardPropagation(self, inputs):
		if len(inputs) != self.n_in-1:
			print 'wrong number of inputs'

		for k in range(self.hidl):
			if k == 0:
				for j in range(self.hid_n[k]-1):
					self.activations[k][j+1] = inputs[j]			## for the bias = 1
			else:
				for j in range(self.hid_n[k]):
					sum1 = 0.0
					for i in range(self.hid_n[k-1]):
						sum1 = sum1 + (self.activations[k-1][i] * self.Weights[k-1][i][j] )
					if k == self.hidl -1:
						self.activations[k][j] = Sigmoid(sum1)
					else:
						self.activations[k][j] = TAN(sum1)


	def backPropagatation(self, target):

		for k in reversed(xrange(self.hidl)):
			nl = k - 1
			if k == self.hidl-1:
				for j in range(self.hid_n[k]):
					for i in range(self.hid_n[k-1]):
						self.deltas[nl][i][j] = (target - self.activations[k][j]) * Sigmoid(self.activations[k][j], True)
			else:
				for j in range(self.hid_n[k]):
					total_e = 0.0
					for i in range(self.hid_n[k+1]):
						total_e = total_e + (self.deltas[nl+1][j][i] * self.Weights[nl+1][j][i])

					for i in range(self.hid_n[k-1]):
						self.deltas[nl][i][j] = TAN(self.activations[k][j], True) * total_e
			if nl==0:
				break

	def UpdateWeights(self, alpha, M):
		for k in range(len(self.Weights)-1):
			for j in range(len(self.Weights[k])):
				for i in range(len(self.Weights[k][j])):
					change = self.deltas[k][j][i]*self.activations[k][j]
					self.Weights[k][j][i] = self.Weights[k][j][i] + (alpha*change) + (M* self.ChangeM[k][j][i])
					self.ChangeM[k][j][i] = change

	def GetClass(self, data):
		self.ForwardPropagation(data)
		eps = 0.5				# threshold to decide on class 1 or class 2
		if self.activations[self.hidl-1][0] < eps:
			t = 0
		else:
			t = 1
		return t
	def testData(self):
		w, h = 400, 400
		eps = 0.5
		data = np.zeros((h, w, 3), dtype=np.uint8)
		x = 0.0
		y = 0.0
		for j in range(h):
			for i in range(w):
				if j < (h/2) and j >= 0 and i >= 0 and i < (w/2):
					x = (i - (w/2)) / 200.0
					y = ((h/2) - j) / 200.0
				elif j < (h/2) and j >= 0 and i >= (w/2) and i <= (w):
					x = (i - (w/2)) / 200.0
					y = ((h/2) - j) / 200.0
				elif j >= (h/2) and j <= h and i >= 0 and i < (w/2):
					x = (i - (h/2)) / 200.0
					y = ((-1*j) + (w/2)) / 200.0
				elif j >= (h/2) and j <= h and i >= (w/2) and i <= (w):
					x = (i - (h/2)) / 200.0
					y = ((-1*j) + (w/2)) / 200.0

				target = self.GetClass([x,y])
				if  target == 0:
					data[i,j] = [0,0,0]
				else:
					data[i,j] = [255,255,255]
		img = Image.fromarray(data, 'RGB')
		img.save('Boundry.png')
		img.show()

	def GetWeights(self):
		for k in range(len(self.Weights)-1):
			for j in range(hid_n[k]):
				for i in range(hid_n[k+1]):
					print k,j,i
					print self.Weights[k][j][i]

	def calculateError(self, target, output):
		error = 0.0
		for i in range(self.n_out):
			error = error + ( 0.5 *(target - output)**2 )
		return error

	def train(self, data, labels, epochs, learning_rate = 0.01 , momentum = 0.1):
		i = 0
		error = 100
		while (i < epochs): #and error > 0.00000001):
			error = 0.0
			PredictedLabels = []
			ccr1 = 0
			ccr2 = 0
			for j in range(len(data)):
				self.ForwardPropagation(data[j])
				self.backPropagatation(labels[j])
				PredictedLabels.append(self.GetClass(data[j]))
				error += self.calculateError(labels[j],PredictedLabels[j])
				self.UpdateWeights(learning_rate, momentum)
				if labels[j] == PredictedLabels[j]:
					if labels[j] == 1:
						ccr1 = ccr1 + 1
					else:
						ccr2 = ccr2 + 1
			error = error/len(data)
			if i%100==0:
				MCCR = min((1.0*ccr1)/c1, (1.0*ccr2)/c2)
				print('error %f' % error, 'ccr1 %f' % ccr1, 'ccr2 %f' % ccr2, 'MCCR %f' % MCCR)
				error_array.append(100.0*error)
				epoch_array.append(i)
			#if i == Max_epochs - 1:
			i = i + 1
		for j in range(len(data)):
			outputLabels.append(PredictedLabels[j])
		self.testData()			# call testData function after training is done


def main():
	circle = '/home/hager/Downloads/circle.txt'
	ring = '/home/hager/Downloads/ring.txt'
	xor = '/home/hager/Downloads/xor.txt'
	spiral = '/home/hager/Downloads/spirals.txt'
	Max_epochs = 3000

	ReadFromFile(ring)		#10,5,2

	x1 = []
	x2 = []
	y1 = []
	y2 = []

	for j in range(len(training_labels)):
	  training_labels[j] -= 1

	for i in range(len(training_data)):
		if(training_labels[i] == 1):
			x1.append(training_data[i][0])
			y1.append(training_data[i][1])
			c1 = c1 + 1
		else:
			x2.append(training_data[i][0])
			y2.append(training_data[i][1])
			c2 = c2 + 1


	hl = 3  			# number of hidden layers
	hn = [5,5,2]			# neurons of each hidden layer

	nn = NeuralNet(2, hl, hn, 1)			# intialize an instance of Neural Net class with certain number of hidden layers
	nn.train(training_data,training_labels, Max_epochs)


	##########Plotting################
	plt.plot(epoch_array, error_array)
	plt.xlabel('number of epochs')
	plt.ylabel('Mean square error %')
	plt.show()


	plt.figure(1)
	plt.rcParams['axes.facecolor']='black'
	plt.subplot(211)
	plt.scatter(x1,y1,color='blue')
	plt.scatter(x2,y2,color='yellow')
	plt.title('Original')


	x1 = []
	x2 = []
	y1 = []
	y2 = []


	for i in range(len(training_data)):
		if(outputLabels[i] == 1):
			x1.append(training_data[i][0])
			y1.append(training_data[i][1])
		else:
			x2.append(training_data[i][0])
			y2.append(training_data[i][1])


	plt.subplot(212)
	plt.scatter(x1,y1,color = 'blue' )#,color='blue')
	plt.scatter(x2,y2, color = 'yellow')
	plt.title('Classification Result')
	plt.show()
	plt.savefig('common_labels.png', dpi=300)

if __name__=='__main__':
	main()
