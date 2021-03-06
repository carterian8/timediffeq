import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import tensorflow as tf
import timediffeq

# Enable tensorflow's eager execution.  This is already done by default in timediffeq so 
# this isn't really necessary...
tf.enable_eager_execution()


# Taken from torchdiffeq's latent_ode.py example: https://github.com/rtqichen/torchdiffeq
def generate_spiral2d(nspiral=1000,
                      ntotal=500,
                      nsample=100,
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.1,
                      a=0.,
                      b=1.,
                      savefig=True):
    """Parametric formula for 2d spiral is `r = a + b * theta`.

    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      nsample: number of sampled datapoints for model fitting per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
      savefig: plot the ground truth for sanity check

    Returns: 
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """

    # add 1 all timestamps to avoid division by 0
    orig_ts = np.linspace(start, stop, num=ntotal)
    samp_ts = orig_ts[:nsample]

    # generate clock-wise and counter clock-wise spirals in observation space
    # with two sets of time-invariant latent dynamics
    zs_cw = stop + 1. - orig_ts
    rs_cw = a + b * 50. / zs_cw
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    orig_traj_cw = np.stack((xs, ys), axis=1)

    zs_cc = orig_ts
    rw_cc = a + b * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=1)

    if savefig:
        plt.figure()
        plt.plot(orig_traj_cw[:, 0], orig_traj_cw[:, 1], label='clock')
        plt.plot(orig_traj_cc[:, 0], orig_traj_cc[:, 1], label='counter clock')
        plt.legend()
        plt.savefig('./ground_truth.png', dpi=500)
        print('Saved ground truth spiral at {}'.format('./ground_truth.png'))

    # sample starting timestamps
    orig_trajs = []
    samp_trajs = []
    for _ in range(nspiral):
        # don't sample t0 very near the start or the end
        t0_idx = npr.multinomial(
            1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        t0_idx = np.argmax(t0_idx) + nsample

        cc = bool(npr.rand() > .5)  # uniformly select rotation
        orig_traj = orig_traj_cc if cc else orig_traj_cw
        orig_trajs.append(orig_traj)

        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)

    return orig_trajs, samp_trajs, orig_ts, samp_ts


if __name__ == '__main__':

	# Generate toy spiral data
	nspiral = 1000
	start = 0.
	stop = 6 * np.pi
	noise_std = .3
	a = 0.
	b = .3
	ntotal = 1000
	nsample = 100
	orig_trajs, samp_trajs, orig_ts, samp_ts = generate_spiral2d(
		nspiral=nspiral,
		start=start,
		stop=stop,
		noise_std=noise_std,
		a=a, b=b
	)

	# Turn the toy data into a tf.data.Dataset
	samp_ts = np.reshape(
		np.repeat(samp_ts, samp_trajs.shape[0]), (nspiral, nsample), order="F"
	)
	dataset = tf.data.Dataset.from_tensor_slices(
		(tf.cast(samp_trajs, dtype=tf.float32), tf.cast(samp_ts, dtype=tf.float32))
	)
	dataset = dataset.batch(nspiral)

	# Create a basic LatentODEVae model.  This includes an RNN encoder, dense ODE model, 
	# and a dense decoder.
	latent_dim = 4
	nhidden = 20
	rnn_nhidden = 25
	obs_dim = 2
	model = timediffeq.BasicLatentODEVae(
		samp_trajs.shape,
		obs_dim,
		latent_dim=latent_dim,
		ode_Dh=nhidden,
		encoder_Dh=rnn_nhidden,
		decoder_Dh=nhidden
	)
	
	# Train the model
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
	timediffeq.train(dataset, model, optimizer, epochs=10)









	