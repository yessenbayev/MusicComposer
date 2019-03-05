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
decoder_input = Input(shape=(120,))
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
ae = Model(input_img, autoenc)


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
    epochs=50,
    batch_size=400, 
    shuffle=True, 
    verbose=1, 
    validation_data=(x_valid, x_valid), 
    # validation_freq=5,
)

# Plot training & validation loss values
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Loss.png')

ae.save('autoencoder.h5')
decoder.save('decoder.h5')
encoder.save('encoder.h5')
np.save('ae_history.npy', train_history.history)
