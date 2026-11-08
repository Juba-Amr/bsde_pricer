
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.interpolate import RegularGridInterpolator


def cn_basket_prices(S0_points, K, T, r, sigma, corr,
                     M=15, N_time=100, S_max_mult=3.0):
    """
    Crank-Nicolson solver for a d-dimensional equal-weight
    European basket call.

    S0_points: array (n_test, d)
    Returns: prices (n_test,)
    """
    X = np.asarray(S0_points, dtype=float)
    if X.ndim == 1:
        X = X[None, :]

    n_test, d = X.shape
    sigma = np.broadcast_to(
        np.asarray(sigma, dtype=float), (d,)
    )
    corr = np.eye(d) if corr is None else np.asarray(corr)

    S_max = S_max_mult * max(np.max(X), K)
    s = np.linspace(0, S_max, M + 1)
    h = s[1] - s[0]
    dt = T / N_time

    m = M + 1
    G = m ** d

    E = eye(m, format="csr")

    # Centered derivative matrices, with no periodic wrapping.
    D1 = diags(
        [-np.ones(m-1), np.ones(m-1)],
        [-1, 1], shape=(m, m), format="csr"
    ) / (2*h)

    D2 = diags(
        [np.ones(m-1), -2*np.ones(m), np.ones(m-1)],
        [-1, 0, 1], shape=(m, m), format="csr"
    ) / h**2

    # Derivative rows on boundaries are not evaluated directly.
    Q1 = diags(s) @ D1
    Q2 = diags(s**2) @ D2

    def tensor_operator(operators):
        out = operators[0]
        for op in operators[1:]:
            out = kron(out, op, format="csr")
        return out

    A = -r * eye(G, format="csr")

    for i in range(d):
        ops = [E] * d
        ops[i] = Q1
        A = A + r * tensor_operator(ops)

        ops = [E] * d
        ops[i] = Q2
        A = A + 0.5 * sigma[i]**2 * tensor_operator(ops)

    for i in range(d):
        for j in range(i+1, d):
            c = corr[i, j] * sigma[i] * sigma[j]
            if c == 0:
                continue
            ops = [E] * d
            ops[i] = Q1
            ops[j] = Q1
            A = A + c * tensor_operator(ops)

    # Initial payoff and upper-boundary mask
    shape = (m,) * d
    basket = np.zeros(shape)
    upper = np.zeros(shape, dtype=bool)

    for i in range(d):
        coord = s.reshape(
            (1,) * i + (m,) + (1,) * (d-i-1)
        )
        basket += coord / d

        face = [slice(None)] * d
        face[i] = M
        upper[tuple(face)] = True

    V = np.maximum(basket - K, 0).ravel()
    basket_flat = basket.ravel()
    upper_flat = upper.ravel()

    # Upper faces: Dirichlet conditions.
    # Lower faces: natural degenerate Black-Scholes operator.
    A = A.tolil()
    A[upper_flat, :] = 0
    A = A.tocsr()

    I = eye(G, format="csr")
    B = (I - 0.5 * dt * A).tocsc()
    C = (I + 0.5 * dt * A).tocsr()

    solver = splu(B)

    for n in range(N_time):
        rhs = C @ V

        tau = (n + 1) * dt
        rhs[upper_flat] = np.maximum(
            basket_flat[upper_flat] - K*np.exp(-r*tau),
            0.0
        )

        V = solver.solve(rhs)

    interpolator = RegularGridInterpolator(
        [s] * d, V.reshape(shape),
        bounds_error=False, fill_value=None
    )

    return np.asarray(interpolator(X)).reshape(-1)
