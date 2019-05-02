import tensorflow as tf
from .neural_ode import NeuralODE


class LatentODEVae(tf.keras.Model):
	"""TODO: Description """
	def __init__(self, rnn_cell, rnn_input_dim, ode_net, decoder):
		
		super(LatentODEVae, self).__init__()
		
		# The encoder net consists of a single RNN layer which evaluates the input
		# sequence backwards
		self.encoder = tf.keras.models.Sequential(
			[
				tf.keras.layers.RNN(rnn_cell, go_backwards=True, input_shape=rnn_input_dim)
			]
		)
		self.ode_net = ode_net
		self.decoder = decoder

	def sample(self, eps=None):
		if eps is None:
			eps = tf.random.normal(shape=(100, self.latent_dim))
		return self.decode(eps, apply_sigmoid=True)

	def encode(self, input):
		# input should be of shape (batch, timesteps, ...) for the encoder RNN.  The input
		# time sequence will be processed backwards...
		
		mean, logvar = tf.split(self.encoder(input), num_or_size_splits=2, axis=1)
		return mean, logvar

	def reparameterize(self, mean, logvar):
		eps = tf.random.normal(shape=mean.shape)
		return eps * tf.exp(logvar * .5) + mean
	
	def neural_ode_forward(self, z0, t):
		neural_ode = NeuralODE(self.ode_net, t=t)
		return neural_ode.forward(z0, return_states="tf")

	def decode(self, z, apply_sigmoid=False):
		logits = self.decoder(z)
		if apply_sigmoid:
		  probs = tf.sigmoid(logits)
		  return probs

		return logits

	def neural_ode_backward(self, outputs, t, dLoss):
		neural_ode = NeuralODE(self.ode_net, t=t)
		return neural_ode.backward(outputs, dLoss)
	
	def encoder_trainable_vars(self):
		return self.encoder.trainable_variables
		
	def neural_ode_trainable_vars(self):
		return self.ode_net.trainable_variables

	def decoder_trainable_vars(self):
		return self.decoder.trainable_variables


class BasicEncoderRnnCell(tf.keras.layers.Layer):

	def __init__(self, units, latent_dim, **kwargs):
		self.units = units
		self.state_size = units
		self.i2h = tf.keras.layers.Dense(self.state_size, activation="tanh")
		self.h2o = tf.keras.layers.Dense(latent_dim * 2)
		super(BasicEncoderRnnCell, self).__init__(**kwargs)
	 
	def call(self, input_at_t, states_at_t):
		prev_out = states_at_t[0]
		h = self.i2h(tf.concat([input_at_t, prev_out], 1))
		out = self.h2o(h)
		return out, h

class BasicODEModel(tf.keras.Model):
	"""TODO: Description """
	def __init__(self, latent_dim, nhidden):
		super(BasicODEModel, self).__init__()
		self.elu = tf.keras.layers.ELU()
		self.linear1 = tf.keras.layers.Dense(nhidden)
		self.linear2 = tf.keras.layers.Dense(nhidden)
		self.linear3 = tf.keras.layers.Dense(latent_dim)

	def call(self, inputs, **kwargs):
		t, y = inputs
		out = self.linear1(y)
		out = self.elu(out)
		out = self.linear2(out)
		out = self.elu(out)
		out = self.linear3(out)
		return out

class BasicLatentODEVae(LatentODEVae):
	"""TODO: Description """
	def __init__(self, rnn_input_dim, out_dim, latent_dim=4, ode_Dh=20, encoder_Dh=25, decoder_Dh=20):
		
		self.latent_dim = latent_dim
		
		# Create basic ode and decoder nets with the latent dimensions, and number of 
		# hidden units given
		
		ode_net = BasicODEModel(latent_dim, ode_Dh)
		decoder = tf.keras.models.Sequential(
			[
				tf.keras.layers.Dense(decoder_Dh),
				tf.keras.layers.ReLU(),
				tf.keras.layers.Dense(out_dim)
			]
		)
		
		# Initialize the latent ode
		super(BasicLatentODEVae, self).__init__(
			BasicEncoderRnnCell(encoder_Dh, latent_dim), rnn_input_dim, ode_net, decoder
		)
