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
             
def nd_grid_price(d, S0, K, T, r, sigma, corr, M=25, N_time=200, S_max_mult=3.0):
    """
    Naive dense tensor-grid explicit finite-difference solver for the
    d-dimensional Black-Scholes basket call PDE. Demonstrates cost growth,
    not meant to be a production-grade solver.

    Returns (price, elapsed_seconds, grid_points, peak_memory_bytes)
    """

    t0 = time.time()
    sigma = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
    if sigma.size == 1 and d > 1:
        sigma = np.full(d, sigma.item())
    if corr is None:
        corr = np.eye(d)
    corr = np.asarray(corr, dtype=np.float64)

    S_max = S_max_mult * max(np.max(S0), K)
    h = S_max / M
    axes = [np.linspace(0, S_max, M + 1) for _ in range(d)]
    mesh = np.meshgrid(*axes, indexing='ij')  # each shape (M+1,)*d

    basket = sum(mesh) / d  # equal-weight basket
    V = np.maximum(basket - K, 0.0).astype(np.float64)

    dt = T / N_time
    sigma = np.asarray(sigma, dtype=np.float64)
    corr = np.asarray(corr, dtype=np.float64)

    grid_points = V.size
    mem_bytes = V.nbytes

    for step in range(N_time):
        L = np.zeros_like(V)
        for i in range(d):
            dV = (np.roll(V, -1, axis=i) - np.roll(V, 1, axis=i)) / (2 * h)
            L += r * mesh[i] * dV
            for j in range(d):
                d2V = (np.roll(np.roll(V, -1, axis=i), -1, axis=j)
                       - np.roll(np.roll(V, -1, axis=i), 1, axis=j)
                       - np.roll(np.roll(V, 1, axis=i), -1, axis=j)
                       + np.roll(np.roll(V, 1, axis=i), 1, axis=j)) / (4 * h * h)
                L += 0.5 * sigma[i] * sigma[j] * corr[i, j] * mesh[i] * mesh[j] * d2V
        L -= r * V
        V = V + dt * L

        # reset boundary (Dirichlet): F=0 at all-zero face, linear payoff at S_max face
        slicer_lo = [slice(None)] * d
        slicer_hi = [slice(None)] * d
        for i in range(d):
            slicer_lo[i] = 0
            V[tuple(slicer_lo)] = 0.0
            slicer_lo[i] = slice(None)
            slicer_hi[i] = M
            V[tuple(slicer_hi)] = np.maximum(sum(mesh)[tuple(slicer_hi)] / d
                                              - K * np.exp(-r * (step + 1) * dt), 0.0)
            slicer_hi[i] = slice(None)

    # interpolate at S0 (nearest-grid-point, simple for demo purposes)
    idx = tuple(int(round(s / h)) for s in S0)
    idx = tuple(min(max(i, 0), M) for i in idx)
    price = V[idx]

    elapsed = time.time() - t0
    return price, elapsed, grid_points, mem_bytes