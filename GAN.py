from keras.layers import LeakyReLU
from keras.layers import Conv2D
import numpy as np
from matplotlib import pyplot
import os
import tensorflow as tf 
from numpy.random import randn
from numpy.random import randint
from numpy import zeros
from numpy import vstack
from numpy import expand_dims
from numpy import ones
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense

from keras.datasets.mnist import load_data
(train_X, train_Y), (test_X, test_Y) = load_data()

for i in range(25):
	pyplot.subplot(5, 5, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(train_X[i], cmap='gray_r')
pyplot.show()

#printing data-labels
print("----------------")
for i in range(25):
    print(train_Y[i])

def discriminator(in_shape=(28,28,1)):
	modal = Sequential()
    # convolution 2D layer addition
	modal.add(Conv2D(64,(3,3),input_shape=in_shape,strides=(2,2),padding='same'))
    # Leaky ReLU addition
	modal.add(LeakyReLU(alpha=0.2))
    # drop-out layer addition
	modal.add(Dropout(0.4))
    # convolution 2D layer addition
	modal.add(Conv2D(64, (3,3),padding='same',strides=(2, 2)))
    # Leaky ReLU addition
	modal.add(LeakyReLU(alpha=0.2))
    # drop-out layer addition
	modal.add(Dropout(0.4))
    # flattening our modal
	modal.add(Flatten())
    # sigmoid activation
	modal.add(Dense(1,activation='sigmoid'))
    # get optimiser
	optimsr = Adam(beta_1=0.5,lr=0.0002)
    # compiling our modal
	modal.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer=optimsr)
	return modal

def makeRealSamples(data,numSample):
	iterx = randint(0,data.shape[0],numSample)
	a = data[iterx]
	b = np.ones((numSample,1))
	return a,b

def loadRealSamples():
	(train_X, _), (_, _) = load_data()
	S = tf.expand_dims(train_X,axis=-1)
	S = np.array(S,dtype=np.float32)
	S = S/255.00
	return S

def savingPlot(examples,num):
	for i in range(num*num):
		pyplot.subplot(num,num,i+1)
		pyplot.axis('off')
		pyplot.imshow(examples[i, :, :, 0],cmap='gray_r')
	pyplot.show()

model = discriminator()
model.summary()

def generator(latent_dimension):
	num_nodes = 128*7*7
    # get a model
	modal = Sequential()
    # add nodes & latent dimension
	modal.add(Dense(num_nodes, input_dim= latent_dimension))
    # Leaky ReLU addition
	modal.add(LeakyReLU(alpha=0.2))
    # Reshaping
	modal.add(Reshape((7,7,128)))
    # convolution 2D layer addition
	modal.add(Conv2DTranspose(128,(4,4),padding='same',strides=(2,2)))
    # Leaky ReLU addition
	modal.add(LeakyReLU(alpha=0.2))
    # convolution 2D layer addition
	modal.add(Conv2DTranspose(128,(4,4),padding='same',strides=(2,2)))
    # Leaky ReLU addition
	modal.add(LeakyReLU(alpha=0.2))
    # convolution 2D layer addition
	modal.add(Conv2D(1,(7,7),activation='sigmoid',padding='same'))
	return modal

def gan(gen_modal,disc_modal):
	disc_modal.trainable = False
    # get model
	modal = Sequential()
    # add generator model 
	modal.add(gen_modal)
    # add discriminator modal
	modal.add(disc_modal)
    # get optimiser
	optm = Adam(lr=0.0002,beta_1=0.5)
    # compiling our modal
	modal.compile(optimizer=optm,loss='binary_crossentropy')
	return modal

def latentPointsGeneration( latent_dimension,numSamples):
    # generating input using random(randn)
	inputX = randn( latent_dimension*numSamples)
    # reshape according to latent dimension
	inputX = inputX.reshape(numSamples,latent_dimension)
	return inputX

def fakeSamplesGeneration(gen_model,latent_dimension,n_samples):
    # getting latent points
	inputX = latentPointsGeneration(latent_dimension,n_samples)
	a = gen_model.predict(inputX)
	b = zeros((n_samples,1))
	return a,b

def performanceSummary(epoch,gen_model,disc_model,data,latent_dimension,n_samples=100):
	realX, realY = makeRealSamples(data,n_samples)
	_, acc_real = disc_model.evaluate(realX,realY,verbose=0)
	fakeX, fakeY = fakeSamplesGeneration(gen_model,latent_dimension,n_samples)
	_, acc_fake = disc_model.evaluate(fakeX, fakeY, verbose=0)
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	savingPlot(fakeX, epoch)
	filename = 'generator_model_%03d.h5' % (epoch + 1)
	gen_model.save(filename)

# function to train our modal
def train_model(gen_model,disc_model,gan_model,latent_dimension,data,numEpochs=1,numBatch=256):
	batchPerEpoch = data.shape[0]//numBatch
	half_batch = numBatch//2
	for i in range(numEpochs):
		for j in range(batchPerEpoch):
            # getting real samples
			realX,realY = makeRealSamples(data,half_batch)
            # getting fake samples
			fakeX,fakeY = fakeSamplesGeneration(gen_model,latent_dimension,half_batch)
			X,y = vstack((realX,fakeX)),vstack((realY,fakeY))
			discrimLoss, _ = disc_model.train_on_batch(X,y)
			X_gan = latentPointsGeneration(latent_dimension,numBatch)
			y_gan = ones((numBatch,1))
			ganLoss = gan_model.train_on_batch(X_gan, y_gan)
			print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1,batchPerEpoch,discrimLoss,ganLoss))
		if (i+1) % 2 == 0:
			performanceSummary(i,gen_model,disc_model,data,latent_dimension)

latent_dimension = 100                    # size of the latent space
disc_model = discriminator()
gen_model = generator( latent_dimension)
gan_model = gan(gen_model, disc_model)
data = loadRealSamples()
train_model(gen_model, disc_model, gan_model,latent_dimension, data)


# # loading model loading & image generation
# model = load_model('generator_model_002.h5',compile = False)
# latent_pts = latentPointsGeneration(100,100)
# X = model.predict(latent_pts)
# savingPlot(X,10)
