import numpy as np
from scipy.linalg import solve_banded
import time


def bergman_fd_nd(S0, K, T, r_l, r_b, sigma, d, M=40, N=200,
                  S_min_mult=0.05, S_max_mult=5.0,
                  picard_tol=1e-8, picard_maxiter=10,
                  rannacher_steps=4, weight=None, corr=None):

    t0 = time.time()

    S0 = np.asarray(S0, dtype=float).reshape(-1)
    if S0.size == 1:
        S0 = np.full(d, S0.item())
    if S0.size != d:
        raise ValueError(f"S0 must have length {d}")

    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    if sigma.size == 1:
        sigma = np.full(d, sigma.item())
    if sigma.size != d:
        raise ValueError(f"sigma must have length {d}")

    if weight is None:
        weight = np.full(d, 1.0 / d)
    else:
        weight = np.asarray(weight, dtype=float).reshape(-1)
        if weight.size != d:
            raise ValueError(f"weight must have length {d}")
        weight = weight / weight.sum()

    if corr is None:
        corr = np.eye(d)
    else:
        corr = np.asarray(corr, dtype=float)
        if corr.shape != (d, d):
            raise ValueError(f"corr must have shape {(d, d)}")

    if not np.allclose(corr, corr.T):
        raise ValueError("corr must be symmetric")

    S_min = max(S_min_mult * min(S0.min(), K), 1e-8)
    S_max = S_max_mult * max(S0.max(), K)

    x = np.linspace(np.log(S_min), np.log(S_max), M + 1)
    h = x[1] - x[0]
    S = np.exp(x)
    dt = T / N

    mesh = np.meshgrid(*([S] * d), indexing="ij")
    basket = sum(weight[i] * mesh[i] for i in range(d))
    V = np.maximum(basket - K, 0.0)
    grid_points = V.size

    cov = corr * np.outer(sigma, sigma)
    bands = []

    for i in range(d):
        a = 0.5 * cov[i, i] / h**2
        conv = r_l - 0.5 * cov[i, i]
        b = conv / (2 * h)
        n = M - 1

        if abs(conv) * h <= cov[i, i]:
            lo = np.full(n, a - b)
            di = np.full(n, -2 * a - r_l / d)
            up = np.full(n, a + b)
        elif conv >= 0:
            bu = conv / h
            lo = np.full(n, a)
            di = np.full(n, -2 * a - bu - r_l / d)
            up = np.full(n, a + bu)
        else:
            bu = conv / h
            lo = np.full(n, a - bu)
            di = np.full(n, -2 * a + bu - r_l / d)
            up = np.full(n, a)

        bands.append((lo, di, up))

    def grad(V, axis):
        Vm = np.moveaxis(V, axis, -1)
        G = np.zeros_like(Vm)
        G[..., 1:M] = (Vm[..., 2:] - Vm[..., :-2]) / (2 * h)
        return np.moveaxis(G, -1, axis)

    def cross_grad(V, i, j):
        Vi = np.moveaxis(V, i, 0)
        Vj = np.moveaxis(Vi, j if j < i else j, 1)
        return (
            Vj.take(indices=range(2, M + 1), axis=1).take(indices=range(2, M + 1), axis=0)
            - Vj.take(indices=range(2, M + 1), axis=1).take(indices=range(0, M - 1), axis=0)
            - Vj.take(indices=range(0, M - 1), axis=1).take(indices=range(2, M + 1), axis=0)
            + Vj.take(indices=range(0, M - 1), axis=1).take(indices=range(0, M - 1), axis=0)
        ) / (4 * h**2)

    def mixed_derivative(V, i, j):
        return (
            np.roll(np.roll(V, -1, axis=i), -1, axis=j)
            - np.roll(np.roll(V, -1, axis=i), 1, axis=j)
            - np.roll(np.roll(V, 1, axis=i), -1, axis=j)
            + np.roll(np.roll(V, 1, axis=i), 1, axis=j)
        ) / (4 * h**2)

    def cross_operator(V):
        out = np.zeros_like(V)
        for i in range(d):
            for j in range(i + 1, d):
                out += cov[i, j] * mixed_derivative(V, i, j)
        return out

    def reaction(V):
        gx = sum(grad(V, i) for i in range(d))
        return (r_b - r_l) * np.maximum(gx - V, 0.0)

    def axis_solve(V, axis, theta, source=None):
        lo, di, up = bands[axis]

        lo_i = -theta * dt * lo
        di_i = 1.0 - theta * dt * di
        up_i = -theta * dt * up

        Vm = np.moveaxis(V, axis, -1)
        shape = Vm.shape
        Vm = Vm.reshape(-1, M + 1)

        rhs = Vm[:, 1:M].copy()
        rhs += (1 - theta) * dt * (
            lo[None, :] * Vm[:, :-2]
            + di[None, :] * Vm[:, 1:-1]
            + up[None, :] * Vm[:, 2:]
        )

        if source is not None:
            Sm = np.moveaxis(source, axis, -1).reshape(-1, M + 1)
            rhs += Sm[:, 1:M]

        ab = np.zeros((3, M - 1))
        ab[0, 1:] = up_i[:-1]
        ab[1, :] = di_i
        ab[2, :-1] = lo_i[1:]

        out = Vm.copy()

        for j in range(rhs.shape[0]):
            out[j, 1:M] = solve_banded((1, 1), ab, rhs[j])

        return np.moveaxis(out.reshape(shape), -1, axis)

    def boundary(V, tau):
        for axis in range(d):
            sl = [slice(None)] * d
            sl[axis] = 0
            V[tuple(sl)] = 0.0

            sl[axis] = M
            V[tuple(sl)] = np.maximum(
                basket[tuple(sl)] - K * np.exp(-r_b * tau), 0.0
            )

        return V

    def step(V, theta):
        R_old = reaction(V)
        C_old = cross_operator(V)
        R_guess = R_old.copy()
        C_guess = C_old.copy()

        for _ in range(picard_maxiter):
            source = dt * (
                (1 - theta) * (R_old + C_old)
                + theta * (R_guess + C_guess)
            )

            V_new = V.copy()

            for axis in range(d):
                V_new = axis_solve(
                    V_new,
                    axis,
                    theta,
                    source if axis == 0 else None
                )

            R_new = reaction(V_new)
            C_new = cross_operator(V_new)

            err_r = np.max(np.abs(R_new - R_guess))
            err_c = np.max(np.abs(C_new - C_guess))

            R_guess = R_new
            C_guess = C_new

            if max(err_r, err_c) < picard_tol:
                return V_new

        return V_new

    for n in range(N):
        theta = 1.0 if n < rannacher_steps else 0.5
        V = step(V, theta)
        V = boundary(V, (n + 1) * dt)

    price = _multilinear_interp(V, x, np.log(S0))

    return float(price), time.time() - t0, grid_points


def _multilinear_interp(V, x, q):
    d = len(q)
    M = len(x) - 1
    h = x[1] - x[0]

    lows = []
    frac = []

    for i in range(d):
        z = np.clip((q[i] - x[0]) / h, 0, M)
        lo = min(int(np.floor(z)), M - 1)
        lows.append(lo)
        frac.append(z - lo)

    value = 0.0

    for c in range(2 ** d):
        idx = []
        w = 1.0

        for i in range(d):
            bit = (c >> i) & 1
            idx.append(lows[i] + bit)
            w *= frac[i] if bit else 1.0 - frac[i]

        value += w * V[tuple(idx)]

    return value