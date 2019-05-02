import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

def log_normal_pdf(sample, mean, logvar, raxis=1):
	log2pi = tf.math.log(2. * np.pi)
	return tf.reduce_sum(
		-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), 
		axis=raxis)

def compute_loss(model, input, t):
	pred_x, qz0_mean, qz0_logvar, z0, outputs, states = model(inputs=[input, t])
	cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_x, labels=input)
	logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2])
	logpz = log_normal_pdf(z0, 0., 0.)
	logqz_x = log_normal_pdf(z0, qz0_mean, qz0_logvar)
	return -tf.reduce_mean(logpx_z + logpz - logqz_x), outputs

def compute_gradients_and_update(model, input, t, optimizer):
	with tf.GradientTape(persistent=True) as tape:
		loss, outputs = compute_loss(model, input, t)
	
	dLoss = tape.gradient(loss, outputs)
	h_start, dfdh0, dWeights = model.neural_ode_backward(outputs, t[0], dLoss)
	optimizer.apply_gradients(zip(dWeights, model.neural_ode_trainable_vars()))

	encoder_gradients = tape.gradient(loss, model.encoder_trainable_vars())
	optimizer.apply_gradients(zip(encoder_gradients, model.encoder_trainable_vars()))

	decoder_gradients = tape.gradient(loss, model.decoder_trainable_vars())
	optimizer.apply_gradients(zip(decoder_gradients, model.decoder_trainable_vars()))

	del tape
	return loss
	

def train(dataset, model, optimizer, epochs=1):
	
	for epoch in range(epochs):
	
		for batch, (input, t) in enumerate(dataset, start=0):
			loss = compute_gradients_and_update(model, input, t, optimizer)
			
			print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch, loss))
