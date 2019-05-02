import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy import integrate
import tensorflow as tf
import timediffeq

def lotka_volterra(X, t=0, a = 1., b = 0.1, c = 1.5, d = 0.75):
	""" Return the growth rate of fox and rabbit populations. """
	return np.array([a*X[0] - b*X[0]*X[1], -c*X[1] + d*b*X[0]*X[1]])

if __name__ == '__main__':
	total_time_pts = 1000
	t = np.linspace(0, 15, total_time_pts)   # time
	y0 = np.array([10, 5])          # initials conditions: 10 rabbits and 5 foxes
	outputs, info_dict = integrate.odeint(
		func=lotka_volterra,
		y0=y0,
		t=t,
		rtol=1e-6,
		atol=1e-6,
		full_output=True,
	)
	
	# Generate noisy samples from ode outputs
	batch_sz = 4000
	val_batch_sz = 1000
	sample_sz = int(total_time_pts / 4)
	noise_std = 0.5
	samps = []
	t0s = []
	for _ in range(batch_sz + val_batch_sz):
		t0_idx = npr.multinomial(
			1, [1. / (total_time_pts - 2. * sample_sz)] * (total_time_pts - int(2 * sample_sz))
		)
		t0_idx = np.argmax(t0_idx) + sample_sz
		t0s.append(t0_idx)
		samp = outputs.T[:, t0_idx:t0_idx + sample_sz].copy()
		samp += npr.randn(*samp.shape) * noise_std
		samps.append(np.transpose(samp))
	samps = np.stack(samps, axis=0)
	
	# Plot for visualization
	rabbits, foxes = outputs.T
	samp_idx = int(npr.uniform(low=0, high=batch_sz-1))
	noisy_rabbits = samps[samp_idx, :, 0]
	noisy_foxes = samps[samp_idx, :, 1]
	samp_ts = t[t0s[samp_idx]:t0s[samp_idx]+sample_sz]
	f1 = plt.figure()
	plt.plot(t, rabbits, 'r-', label='Rabbits')
	plt.plot(t, foxes  , 'b-', label='Foxes')
	plt.scatter(samp_ts, noisy_rabbits, s=1.0, label='Noisy Rabbits')
	plt.scatter(samp_ts, noisy_foxes, c='g', s=1.0, label='Noisy Foxes')
	plt.grid()
	plt.legend(loc='best')
	plt.xlabel('time')
	plt.ylabel('population')
	f1.savefig('rabbits_and_foxes_1.png', dpi=500)
	
	# Turn the toy data into a tf.data.Dataset
	samp_ts = np.reshape(
		np.repeat(t[:sample_sz], samps.shape[0]), (batch_sz+val_batch_sz, sample_sz), order="F"
	)
	dataset = tf.data.Dataset.from_tensor_slices(
		(tf.cast(samps, dtype=tf.float32), tf.cast(samp_ts, dtype=tf.float32))
	)
	dataset = dataset.batch(batch_sz)
	
	# Create a basic LatentODEVae model.  This includes an RNN encoder, dense ODE model, 
	# and a dense decoder.
	latent_dim = 4
	nhidden = 20
	rnn_nhidden = 25
	obs_dim = 2
	model = timediffeq.BasicLatentODEVae(
		samps.shape,
		obs_dim,
		latent_dim=latent_dim,
		ode_Dh=nhidden,
		encoder_Dh=rnn_nhidden,
		decoder_Dh=nhidden
	)
	
	# Train the model
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
	timediffeq.train(dataset, model, optimizer, epochs=10)
	