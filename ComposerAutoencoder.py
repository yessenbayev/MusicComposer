import numpy as np
import os
import keras
import midi
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, Permute, RepeatVector, ActivityRegularization, TimeDistributed, Lambda, SpatialDropout1D
from keras.layers.convolutional import Conv1D, Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.embeddings import Embedding
from keras.layers.local import LocallyConnected2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.initializers import RandomNormal
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import plot_model
from keras import backend as K
from keras import regularizers
from keras.engine.topology import Layer
K.set_image_data_format('channels_first')

class ComposerAutoencoder:
    def __init__(self):
        NUM_EPOCHS = 2000
        LR = 0.001
        CONTINUE_TRAIN = False
        PLAY_ONLY = False
        USE_EMBEDDING = False
        USE_VAE = False
        WRITE_HISTORY = True
        NUM_RAND_SONGS = 10
        DO_RATE = 0.1
        BN_M = 0.9
        VAE_B1 = 0.02
        VAE_B2 = 0.1

        BATCH_SIZE = 350
        MAX_LENGTH = 16
        PARAM_SIZE = 120
        NUM_OFFSETS = 16 if USE_EMBEDDING else 1


        print("Loading Data...")
        y_samples = np.load('samples.npy')
        y_lengths = np.load('lengths.npy')
        num_samples = y_samples.shape[0]
        num_songs = y_lengths.shape[0]
        print("Loaded " + str(num_samples) + " samples from " + str(num_songs) + " songs.")
        print(np.sum(y_lengths))
        assert(np.sum(y_lengths) == num_samples)

        print("Padding Songs...")
        x_shape = (num_songs * NUM_OFFSETS, 1)
        y_shape = (num_songs * NUM_OFFSETS, MAX_LENGTH) + y_samples.shape[1:]
        x_orig = np.expand_dims(np.arange(x_shape[0]), axis=-1)
        y_orig = np.zeros(y_shape, dtype=y_samples.dtype)
        cur_ix = 0
        for i in range(num_songs):
            for ofs in range(NUM_OFFSETS):
                ix = i*NUM_OFFSETS + ofs
                end_ix = cur_ix + y_lengths[i]
                for j in range(MAX_LENGTH):
                    k = (j + ofs) % (end_ix - cur_ix)
                    y_orig[ix,j] = y_samples[cur_ix + k]
            cur_ix = end_ix
        assert(end_ix == num_samples)
        x_train = np.copy(x_orig)
        y_train = np.copy(y_orig)

        def to_song(encoded_output):
	        return np.squeeze(decoder([np.round(encoded_output), 0])[0])

        def reg_mean_std(x):
            s = K.log(K.sum(x * x))
            return s*s

        def vae_sampling(args):
            z_mean, z_log_sigma_sq = args
            epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.0, stddev=VAE_B1)
            return z_mean + K.exp(z_log_sigma_sq * 0.5) * epsilon

        def vae_loss(x, x_decoded_mean):
            xent_loss = binary_crossentropy(x, x_decoded_mean)
            kl_loss = VAE_B2 * K.mean(1 + z_log_sigma_sq - K.square(z_mean) - K.exp(z_log_sigma_sq), axis=None)
            return xent_loss - kl_loss

        if CONTINUE_TRAIN or PLAY_ONLY:
            print("Loading Model...")
            model = load_model('model.h5', custom_objects=custom_objects)
        else:
            print("Building Model...")

            if USE_EMBEDDING:
                x_in = Input(shape=x_shape[1:])
                print((None,) + x_shape[1:])
                x = Embedding(x_train.shape[0], PARAM_SIZE, input_length=1)(x_in)
                x = Flatten(name='pre_encoder')(x)
            else:
                x_in = Input(shape=y_shape[1:])
                print((None,) + y_shape[1:])
                x = Reshape((y_shape[1], -1))(x_in)
                print(K.int_shape(x))
                
                x = TimeDistributed(Dense(2000, activation='relu'))(x)
                print(K.int_shape(x))
                
                x = TimeDistributed(Dense(200, activation='relu'))(x)
                print(K.int_shape(x))

                x = Flatten()(x)
                print(K.int_shape(x))

                x = Dense(1600, activation='relu')(x)
                print(K.int_shape(x))
                
                if USE_VAE:
                    z_mean = Dense(PARAM_SIZE)(x)
                    z_log_sigma_sq = Dense(PARAM_SIZE)(x)
                    x = Lambda(vae_sampling, output_shape=(PARAM_SIZE,), name='pre_encoder')([z_mean, z_log_sigma_sq])
                else:
                    x = Dense(PARAM_SIZE)(x)
                    x = BatchNormalization(momentum=BN_M, name='pre_encoder')(x)
            print(K.int_shape(x))
            
            x = Dense(1600, name='encoder')(x)
            x = BatchNormalization(momentum=BN_M)(x)
            x = Activation('relu')(x)
            if DO_RATE > 0:
                x = Dropout(DO_RATE)(x)
            print(K.int_shape(x))

            x = Dense(MAX_LENGTH * 200)(x)
            print(K.int_shape(x))
            x = Reshape((MAX_LENGTH, 200))(x)
            x = TimeDistributed(BatchNormalization(momentum=BN_M))(x)
            x = Activation('relu')(x)
            if DO_RATE > 0:
                x = Dropout(DO_RATE)(x)
            print(K.int_shape(x))

            x = TimeDistributed(Dense(2000))(x)
            x = TimeDistributed(BatchNormalization(momentum=BN_M))(x)
            x = Activation('relu')(x)
            if DO_RATE > 0:
                x = Dropout(DO_RATE)(x)
            print(K.int_shape(x))

            x = TimeDistributed(Dense(y_shape[2] * y_shape[3], activation='sigmoid'))(x)
            print(K.int_shape(x))
            x = Reshape((y_shape[1], y_shape[2], y_shape[3]))(x)
            print(K.int_shape(x))
            
            if USE_VAE:
                model = Model(x_in, x)
                model.compile(optimizer=Adam(lr=LR), loss=vae_loss)
            else:
                model = Model(x_in, x)
                model.compile(optimizer=RMSprop(lr=LR), loss='binary_crossentropy')

            plot_model(model, to_file='model.png', show_shapes=True)

if __name__=="__main__":
    autoencoder = ComposerAutoencoder()