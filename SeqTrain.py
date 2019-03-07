import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from collections import deque
from DataGenerator import MidiDataGenerator


def train(directory, prefix='default', m=16, epoch=100):
	class SaveModelandEval(keras.callbacks.Callback):
		def on_epoch_end(self, epoch, logs):
			if epoch+1 not in [10, 20, 50, 100, 150, 200, 250]: return 
			
			prefixn = "_".join([prefix, str(epoch+1), ])
			# ae.save(prefixn+'_autoencoder.h5')
			decoder.save(prefixn+'_decoder.h5')
			encoder.save(prefixn+'_encoder.h5')


	midi = MidiDataGenerator(directory, m=m)
	xtrain = midi.samples(all=True)
	xtrain = xtrain if xtrain.shape[0] <= 20000 else xtrain[:20000]

	x_shape = xtrain.shape
	print('Training from %s'%(directory))
	print('Sample Size for Training: ', x_shape)

	# Encoder
	input_img  = keras.layers.Input(shape=x_shape[1:])
	encoded = keras.layers.Reshape(target_shape=(x_shape[1], x_shape[2]*x_shape[3]))(input_img)
	encoded = keras.layers.TimeDistributed(keras.layers.Dense(2000, activation='relu'))(encoded)
	encoded = keras.layers.TimeDistributed(keras.layers.Dense(200, activation='relu'))(encoded)
	encoded = keras.layers.Flatten()(encoded)
	encoded = keras.layers.Dense(1600, activation='relu')(encoded)

	encoded = keras.layers.Dense(120)(encoded)
	encoded = keras.layers.BatchNormalization(momentum=0.9)(encoded)
	encoder = keras.models.Model(input_img, encoded)

	# Decoder
	decoder_input = keras.layers.Input(shape=(120,))
	decoded = keras.layers.Dense(1600, name='encoder')(decoder_input)
	decoded = keras.layers.BatchNormalization(momentum=0.9)(decoded)
	decoded = keras.layers.Activation('relu')(decoded)

	decoded = keras.layers.Dense(16*200)(decoded)
	decoded = keras.layers.Reshape(target_shape=(16, 200))(decoded)
	decoded = keras.layers.TimeDistributed(keras.layers.BatchNormalization(momentum=0.9))(decoded)
	decoded = keras.layers.Activation('relu')(decoded)

	decoded = keras.layers.TimeDistributed(keras.layers.Dense(2000))(decoded)
	decoded = keras.layers.TimeDistributed(keras.layers.BatchNormalization(momentum=0.9))(decoded)
	decoded = keras.layers.Activation('relu')(decoded)

	decoded = keras.layers.TimeDistributed(keras.layers.Dense(x_shape[2]*x_shape[3], activation='sigmoid'))(decoded)
	decoded = keras.layers.Reshape(target_shape=(x_shape[1], x_shape[2], x_shape[3]))(decoded)
	decoder = keras.models.Model(decoder_input, decoded)

	autoenc = decoder(encoder(input_img))
	ae = keras.Model(input_img, autoenc)
	ae.compile(
		optimizer=keras.optimizers.RMSprop(lr=1e-3), 
		loss='binary_crossentropy', 
	)
	train_history = ae.fit(
		xtrain, 
		xtrain, 
		epochs=epoch,
		batch_size=200, 
		shuffle=True, 
		verbose=1,
		callbacks=[SaveModelandEval()]
	)
	np.save(prefix+'_history.npy', train_history.history)



if __name__ == '__main__': 
	tasks = open(sys.argv[1], 'r').read().split('\n')

	print('Found %d Dataset for Training...'%(len(tasks)))

	for task in tasks:
		train(task, prefix=task.split('/')[-1], m=16, epoch=250)

	print('done')
		