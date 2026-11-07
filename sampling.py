import numpy as np
import tensorflow as tf
from scipy.stats import norm

#default constants
r = 0.02
sigma = 0.2
T = 1


def call_payoff(S, K):
    return tf.maximum(S - K, 0.0)

def brownian_path(num_paths, size, T=T):
    dt = T/(size+1)
    B = np.array(np.random.normal(0,np.sqrt(dt), (num_paths, size)), dtype=np.float32)
    W0 = np.zeros(shape=(num_paths,1), dtype=np.float32)
    W = np.concatenate([W0 ,np.cumsum(B, axis=1, dtype=np.float32)], axis=1, dtype=np.float32)
    return W

def S_path(S0, W_path, r=r, sigma=sigma, T=T, chunk_size=None, full_path=True):
    if not isinstance(S0, (int, float)):
        assert len(S0) == W_path.shape[0], "S0 and W_path must have the same number of paths"
    num_paths, size = W_path.shape
    dt = T / size

    W_lagged = np.concatenate([np.zeros((num_paths, 1)), W_path[:, :-1]], axis=1, dtype=np.float32)
    dW = (W_path - W_lagged)[:, 1:]          

    S0 = np.broadcast_to(np.asarray(S0, dtype=np.float32), (num_paths,)).copy()

    if chunk_size is None:
        chunk_size = size - 1

    if full_path:
        S = np.empty((num_paths, size), dtype=np.float32)
        S[:, 0] = S0
        current = S0.copy()
        for start in range(0, size - 1, chunk_size):
            end = min(start + chunk_size, size - 1)
            dW_chunk = dW[:, start:end]                     
            growth_chunk = 1 + r * dt + sigma * dW_chunk
            cum_chunk = np.cumprod(growth_chunk, axis=1)
            S[:, start+1:end+1] = current[:, None] * cum_chunk
            current = S[:, end]                             
        return S, S[:, -1]
    else:
        S = S0.copy()
        for start in range(0, size - 1, chunk_size):
            end = min(start + chunk_size, size - 1)
            dW_chunk = dW[:, start:end]
            growth_chunk = 1 + r * dt + sigma * dW_chunk
            S *= np.prod(growth_chunk, axis=1)
        return S

def brownian_increment(batch_size, N, d, T, corr=None):
    dt = T / N
    if corr is None:
        return tf.random.normal((batch_size, N, d), stddev=tf.sqrt(dt))
    else:
        L = tf.linalg.cholesky(tf.cast(corr, tf.float32))
        Z = tf.random.normal((batch_size, N, d), stddev=tf.sqrt(dt))
        return tf.einsum('ij,bnj->bni', L, Z)

def eu_call_price(St, K, time_to_maturity=T, r=r, sigma=sigma):
    d1 = (np.log(St/K)+ time_to_maturity*(r + (sigma**2)/2))/(sigma*np.sqrt(time_to_maturity))
    d2 = d1 - (sigma*np.sqrt(time_to_maturity))

    F = norm.cdf
    price = St*F(d1) - np.exp(-r*time_to_maturity)*K*F(d2)
    return price

def monte_carlo_pricer(S0, num_paths, num_steps, K, time_to_maturity, chunk_size=1000):
    S0_vector = np.array([S0]*num_paths)
    W = brownian_path(num_paths=num_paths, size=num_steps)
    S_terminal = S_path(S0=S0_vector, W_path=W, full_path=False, chunk_size=chunk_size)
    payoff = call_payoff(S=S_terminal, K=K)

    outcomes = np.exp(-r*time_to_maturity)*payoff
    return np.mean(outcomes)

def basket_call_payoff(S, K, weights=None):
    d = S.shape[0]
    w = weights if weights is not None else tf.ones((d,), dtype=S.dtype) / d
    basket = tf.reduce_sum(w * S, axis=1, keepdims=True)
    return tf.nn.relu(basket - K)

def y0_init(S0, K, T, batch_size=100000, r=r, sigma=sigma, weights=None, corr=None):
    d = len(S0)
    Z = tf.random.normal((batch_size, d))
    if corr is not None:
        L = tf.linalg.cholesky(tf.cast(corr, tf.float32))
        Z = tf.einsum('bd,ed->be', Z, L)
    sigma = tf.cast(sigma, tf.float32)
    T = tf.cast(T, tf.float32)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * tf.sqrt(T) * Z
    ST = S0 * tf.exp(drift + diffusion)
    payoffs = basket_call_payoff(ST, K, weights)
    discounted = tf.exp(-r * T) * payoffs
    mean = tf.reduce_mean(discounted)
    std_err = tf.math.reduce_std(discounted) / tf.sqrt(float(batch_size))
    return mean, std_err