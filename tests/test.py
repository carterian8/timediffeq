import numpy as np
import scipy
import tensorflow as tf

tf.enable_eager_execution()

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


def lotka_volterra(X, t=0, a = 1., b = 0.1, c = 1.5, d = 0.75):
    """ Return the growth rate of fox and rabbit populations. """
    return np.array([a*X[0] - b*X[0]*X[1], -c*X[1] + d*b*X[0]*X[1]])

total_time_pts = 1000
forward_t = np.linspace(1, 2, 1)   # time
#forward_t = np.linspace(5, 10, int(total_time_pts/2))   # time
backward_t = np.linspace(5, 0, int(total_time_pts/2))
y0 = np.array([10, 5])          # initials conditions: 10 rabbits and 5 foxes

forward_y = scipy.integrate.odeint(
    lotka_volterra,
    y0,
    t=forward_t,
    rtol=1e-6,
    atol=1e-6
)

backward_y = scipy.integrate.odeint(
    lotka_volterra,
    y0,
    t=backward_t,
    rtol=1e-6,
    atol=1e-6
)

print(forward_y.shape)
print(backward_y.shape)

class NeuralOde():
    
    def __init__():
        self.func = func
        self.t = None
        self.rtol = 1e-6
        self.atol = 1e-6
        self.method = 'dopri5'
        
    def _forward(y0):
        with tf.name_scope("forward_odeint"):
            t = tf.cast(self.t, dtype=tf.float32)
            
            outputs, info_dict = tf.contrib.integrate.odeint(
                func=lambda _y, _t: self.func(inputs=(_t, _y)),
                y0=y0,
                t=t,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method,
                full_output=True,
            )
        
        return outputs
    
    #func, y0, t, z_t1, loss_grads
    def _backward(*grad_ys):
        
        # Define the augmented dynamics for the reverse-mode derivative...
        def aug_dynamics(state, t):
            ht = state[0]
            at = -state[1]

            with tf.GradientTape() as tape:
                tape.watch(ht)
                ht_new = self.func(inputs=[t, ht])

            gradients = tape.gradient(
                target=ht_new, sources=[ht] + self.func.weights,
                output_gradients=at
            )

            # return [1.0, ht_new, *gradients]
        
        # Break the reverse-mode derivative into a sequence of separate solves, one 
        # between each consecutive pair of output times...
        z_t1 = # Solve for the final state using the model
        grad_weights = [tf.zeros_like(w) for w in self.func.weights]
        state = [z_t1, grad_ys, grad_weights]
        
        for _t in self.t[::-1]:
            state = scipy.integrate.odeint(
                aug_dynamics,
                state,
                t=_t,
                rtol=1e-6,
                atol=1e-6
            )
            
            # TODO() Adjust the adjoint in the direction of the corresponding partial derivative:
            # dL/dzt_i...
            
            
    @tf.custom_gradient
    def _odeint_adjoint(y0):
        return self._forward(y0), self._backward
    
    def odeint_adjoint(func, y0, t, rtol=1e-6, atol=1e-6, method='dopri5'):
        self.t = t
        self.rtol = rtol
        self.atol = atol
        self.method = method
        return _odeint_adjoint(y0)

