from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

config = {}
config['input_dim'] = 784
config['layers'] = [64, 32]
config['activation'] = 'relu'
config['output_activation'] = 'sigmoid'


class Autoencoder:
  def __init__(self, config):
    encoding_dim = config['layers'][-1]
    # this is our input placeholder
    input = Input(shape=(config['input_dim'],))
    
    # "encoded" is the encoded representation of the input
    print('Encoded')
    encoded = Dense(config['layers'][0], activation=config['activation'])(input)
    print(config['layers'][0], config['activation'])
    for n_nodes in config['layers'][1:]:
      print(n_nodes, config['activation'])
      encoded = Dense(n_nodes, activation=config['activation'])(encoded)
    
    # "decoded" is the lossy reconstruction of the input
    print('Decoded')
    encoded_input = Input(shape=(encoding_dim,))
    if len(config['layers']) > 1:
      decoded = Dense(config['layers'][-2], activation=config['activation'])(encoded)
      decoded_output = Dense(config['layers'][-2], activation=config['activation'])(encoded_input)
      print(config['layers'][-2], config['activation'])
      for n_nodes in reversed(config['layers'][:-2]):
        print(n_nodes, config['activation'])
        decoded = Dense(n_nodes, activation=config['activation'])(decoded)
        decoded_output = Dense(n_nodes, activation=config['activation'])(decoded_output)
      decoded = Dense(config['input_dim'], activation=config['output_activation'])(decoded)
      decoded_output = Dense(config['input_dim'], activation=config['output_activation'])(decoded_output)
      print(config['input_dim'], config['output_activation'])
    else:
      decoded = Dense(config['input_dim'], activation=config['output_activation'])(encoded)
      decoded_output = Dense(config['input_dim'], activation=config['output_activation'])(encoded_input)
      print(config['input_dim'], config['output_activation'])


    
    #### Create Models ####
    # Autoencoder model maps an input to its reconstruction
    self.autoencoder = Model(input, decoded)
    # Encoder model maps an input to its encoded representation
    self.encoder = Model(input, encoded)
    # Decoder model maps an encoded input to a reconstructed output
    self.decoder = Model(encoded_input, decoded_output)

    #### Configure Model
    self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

  def train(self, x_train, x_test, n_epoch, n_batch):
    self.autoencoder.fit(x_train, x_train,
                         epochs=n_epoch,
                         batch_size=n_batch,
                         shuffle=True,
                         validation_data=(x_test, x_test))
  
  def encode(self,x_test):
    encoded_output = self.encoder.predict(x_test)
    return encoded_output
  
  def decode(self,encoded_output):
    decoded_output = self.decoder.predict(encoded_output)
    return decoded_output

if __name__ == "__main__":
  # Import Data
  (x_train, _), (x_test, _) = mnist.load_data()
  x_train = x_train.astype('float32') / 255.
  x_test = x_test.astype('float32') / 255.
  x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
  x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

  # Create Model
  model = Autoencoder(config)

  # Train Model
  n_epoch = 50
  n_batch = 256
  model.train(x_train, x_test, n_epoch, n_batch)

  # Get Results
  encoded_output = model.encode(x_test)
  decoded_output = model.decode(encoded_output)

  # Show Results
  n = 10  # how many digits we will display
  plt.figure(figsize=(20, 4))
  for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.show()


