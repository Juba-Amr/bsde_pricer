import tensorflow as tf
import numpy as np
import time 

def brownian_increment(batch_size, N, d, T, corr=None):
    dt = T / N
    if corr is None:
        return tf.random.normal((batch_size, N, d), stddev=tf.sqrt(dt))
    else:
        L = tf.linalg.cholesky(tf.cast(corr, tf.float32))
        Z = tf.random.normal((batch_size, N, d), stddev=tf.sqrt(dt))
        return tf.einsum('ij,bnj->bni', L, Z)

class SubNet(tf.keras.layers.Layer):
 
    def __init__(self, dim, num_hidden_layers=3, num_units=10):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.dense_layers = [
            tf.keras.layers.Dense(num_units + dim, use_bias=False)
            for _ in range(num_hidden_layers - 1)
        ] + [tf.keras.layers.Dense(dim, use_bias=False)]
        self.bn_layers = [
            tf.keras.layers.BatchNormalization() for _ in range(num_hidden_layers + 1)
        ]
 
    def call(self, x, training=False):
        x = self.bn_layers[0](x, training=training)
        for i in range(self.num_hidden_layers - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i + 1](x, training=training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x, training=training)
        return x
 
 
class DeepBSDE(tf.keras.Model):

    def __init__(self, d, N, X0_low, X0_high, payoff_fn,
                 mu_fn, sigma_fn, f_fn, num_units=10, num_hidden_layers=3,
                 T=1.0, rho=None, verbose=1):
        super().__init__()
        self.subnets = [SubNet(dim=d, num_hidden_layers=num_hidden_layers, num_units=num_units) for _ in range(N - 1)]
        self.verbose = verbose

        # y0 and z0 are now NETWORKS of x, not free scalars/vectors
        self.y0_net = SubNet(dim=1, num_hidden_layers=num_hidden_layers, num_units=num_units)
        self.z0_net = SubNet(dim=d, num_hidden_layers=num_hidden_layers, num_units=num_units)

        # range to sample X0 from, instead of a fixed constant
        self.X0_low = tf.constant(X0_low, dtype=tf.float32)
        self.X0_high = tf.constant(X0_high, dtype=tf.float32)

        self.mu_fn = mu_fn
        self.sigma_fn = sigma_fn
        self.f_fn = f_fn
        self.payoff_fn = payoff_fn

        self.rho = None if rho is None else tf.cast(rho, tf.float32)

        self.d, self.N, self.T = d, N, T
        self.dt = T / N

    def sample_X0(self, batch_size):
        # uniform sampling in a range around your S0 of interest;
        # swap for lognormal / other distribution if you prefer
        u = tf.random.uniform((batch_size, self.d))
        return self.X0_low + u * (self.X0_high - self.X0_low)

    def call(self, dW, training=False, X0=None):
        batch_size = tf.shape(dW)[0]

        if X0 is None:
            X0 = self.sample_X0(batch_size)   # random X0 per batch

        X = X0
        Y = self.y0_net(X0, training=training)          # (batch, 1)
        Z = self.z0_net(X0, training=training)           # (batch, d)

        for n in range(self.N):
            t_n = tf.constant(n * self.dt, dtype=tf.float32)
            dW_n = dW[:, n, :]

            f_n = self.f_fn(t_n, X, Y, Z)
            Y = Y - f_n * self.dt + tf.reduce_sum(Z * dW_n, axis=1, keepdims=True)

            mu_n = self.mu_fn(t_n, X)
            sigma_n = self.sigma_fn(t_n, X)
            X = X + mu_n * self.dt + sigma_n * dW_n

            if n < self.N - 1:
                Z = self.subnets[n](X, training=training)

        return X0, X, Y   # return X0 too, since payoff/eval needs to know starting point context if desired

    @tf.function
    def train_step(self, dW, optimizer):
        with tf.GradientTape() as tape:
            X0, X, Y = self(dW, training=True)
            payoff = self.payoff_fn(X)
            loss = tf.reduce_mean((Y - payoff) ** 2)

        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    def train(self, optimizer=None, num_epochs=1000, batch_size=64, learning_rate=1.0):
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate, decay_steps=num_epochs,
        )
        optimizer = tf.keras.optimizers.Adam(lr_schedule) if optimizer is None else optimizer(lr_schedule)
        
        # Initialize network layers in eager mode
        dummy_dW = tf.zeros(
            (batch_size, self.N, self.d),
            dtype=tf.float32
        )

        _ = self(dummy_dW, training=False)

        # Check BatchNormalization initialization
        bn = self.y0_net.bn_layers[0]
        print("Built:", bn.built)

        for epoch in range(num_epochs):
            dW = brownian_increment(batch_size, self.N, self.d, self.T, corr=self.rho)
            loss = self.train_step(dW, optimizer)

            if self.verbose and epoch % 500 == 0:
                tf.print("Epoch", epoch, "Loss:", loss)

    def predict_price(self, S0_points):
        """
        Evaluate u(0, x) at one or many points.
        S0_points: shape (d,) for a single point, or (batch, d) for many points.
        """
        x = tf.constant(S0_points, dtype=tf.float32)
        if x.shape.rank == 1:
            
            if self.d == 1:
                x = x[:, None]      
            else:
                x = x[None, :]      
        return self.y0_net(x, training=False)[:, 0]
             
