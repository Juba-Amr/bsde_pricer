import numpy as np
import time


def nd_grid_price_funding(S0, T, sigma, corr=None, r_l=0.0, r_b=0.0,
                          payoff=None,M=25,N_time=200,S_max_mult=3.0):
    t0 = time.time()

    S0 = np.asarray(S0, dtype=np.float64).reshape(-1)
    d = S0.size

    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.ndim == 0:
        sigma = np.full(d, sigma)

    if corr is None:
        corr = np.eye(d)
    corr = np.asarray(corr, dtype=np.float64)

    if payoff is None:
        raise ValueError("payoff must be provided")

    if sigma.shape != (d,):
        raise ValueError("sigma must have shape (d,)")

    if corr.shape != (d, d):
        raise ValueError("corr must have shape (d,d)")

    S_max = S_max_mult * max(np.max(S0), 1e-12)
    h = S_max / M

    axes = [np.linspace(0, S_max, M + 1) for _ in range(d)]
    mesh = np.meshgrid(*axes, indexing="ij")
    S_grid = np.stack(mesh, axis=-1)

    V = np.array(payoff(S_grid), dtype=np.float64, copy=True)
    if V.shape != (M + 1,) * d:
        raise ValueError("payoff returned an array with the wrong shape")

    covariance = np.outer(sigma, sigma) * corr
    dt = T / N_time

    grid_points = V.size
    mem_bytes = V.nbytes

    for _ in range(N_time):
        L = np.zeros_like(V)
        delta = []

        for i in range(d):
            dV = (
                np.roll(V, -1, axis=i) -
                np.roll(V, 1, axis=i)
            ) / (2 * h)

            delta.append(dV)
            L += r_l * mesh[i] * dV

        portfolio_cash = V.copy()

        for i in range(d):
            portfolio_cash -= mesh[i] * delta[i]

        L += (r_b - r_l) * np.maximum(-portfolio_cash, 0.0)

        for i in range(d):
            for j in range(d):
                d2V = (
                    np.roll(np.roll(V, -1, axis=i), -1, axis=j)
                    - np.roll(np.roll(V, -1, axis=i), 1, axis=j)
                    - np.roll(np.roll(V, 1, axis=i), -1, axis=j)
                    + np.roll(np.roll(V, 1, axis=i), 1, axis=j)
                ) / (4 * h * h)

                L += (
                    0.5
                    * covariance[i, j]
                    * mesh[i]
                    * mesh[j]
                    * d2V
                )

        L -= r_l * V
        V += dt * L

        for i in range(d):
            lo = [slice(None)] * d
            lo[i] = 0
            lo_inner = [slice(None)] * d
            lo_inner[i] = 1
            V[tuple(lo)] = V[tuple(lo_inner)]

            hi = [slice(None)] * d
            hi[i] = M
            hi_1 = [slice(None)] * d
            hi_1[i] = M - 1
            hi_2 = [slice(None)] * d
            hi_2[i] = M - 2

            V[tuple(hi)] = (
                2 * V[tuple(hi_1)] -
                V[tuple(hi_2)]
            )

    idx = tuple(np.clip(np.round(S0 / h).astype(int), 0, M))
    price = float(V[idx])

    return price, time.time() - t0, grid_points, mem_bytes