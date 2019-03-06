import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from DataGenerator import MidiDataGenerator

print('Is GPU available: ')
print(tf.test.is_gpu_available())

midi = MidiDataGenerator('./raw', m=16)
x_train, pos1 = midi.samples(size=20000)
x_valid, pos2 = midi.samples(size=5000)
x_test,  pos3 = midi.samples(size=5000)

print('Sample Size for Training: ', x_train.shape)
print('Sample Size for Validation: ', x_valid.shape)
print('Sample Size for Testing: ', x_test.shape)


# ============================================================ #
#     Classical Autoencoder, With Time Distribution Layers     #
# ============================================================ #
x_shape = x_train.shape

x = keras.layers.Input(shape=x_shape[1:])

y = keras.layers.Reshape(target_shape=(x_shape[1], x_shape[2]*x_shape[3]))(x)
y = keras.layers.TimeDistributed(keras.layers.Dense(2000, activation='relu'))(y)
y = keras.layers.TimeDistributed(keras.layers.Dense(200, activation='relu'))(y)
y = keras.layers.Flatten()(y)
y = keras.layers.Dense(1600, activation='relu')(y)

y = keras.layers.Dense(120)(y)
y = keras.layers.BatchNormalization(momentum=0.9)(y)

y = keras.layers.Dense(1600, name='encoder')(y)
y = keras.layers.BatchNormalization(momentum=0.9)(y)
y = keras.layers.Activation('relu')(y)

y = keras.layers.Dense(16*200)(y)
y = keras.layers.Reshape(target_shape=(16, 200))(y)
y = keras.layers.TimeDistributed(keras.layers.BatchNormalization(momentum=0.9))(y)
y = keras.layers.Activation('relu')(y)

y = keras.layers.TimeDistributed(keras.layers.Dense(2000))(y)
y = keras.layers.TimeDistributed(keras.layers.BatchNormalization(momentum=0.9))(y)
y = keras.layers.Activation('relu')(y)

y = keras.layers.TimeDistributed(keras.layers.Dense(x_shape[2]*x_shape[3], activation='sigmoid'))(y)
y = keras.layers.Reshape(target_shape=(x_shape[1], x_shape[2], x_shape[3]))(y)

print(x)
print(y)

# autoencorder
ae = keras.Model(x, y)
ae.compile(
	optimizer=keras.optimizers.RMSprop(lr=1e-3), 
	loss='binary_crossentropy', 
)
train_history = ae.fit(
	x_train, 
	x_train, 
	epochs=20,
	batch_size=1, 
	shuffle=True, 
	verbose=1, 
	validation_data=(x_valid, x_valid), 
	# validation_freq=5,
)

ae.save('ae.h5')
np.save('ae_history.npy', train_history.history)