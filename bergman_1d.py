import numpy as np
from scipy.linalg import solve_banded


# ==================================================================
# Boundary condition factories for standard payoffs
# ==================================================================

def call_boundaries(K, r_b):
    """
    Dirichlet boundaries for a vanilla call.
    Hedge requires borrowing near/above K -> discount at r_b.
    """
    left_bc_fn = lambda tau, S: 0.0
    right_bc_fn = lambda tau, S: max(S - K * np.exp(-r_b * tau), 0.0)
    return left_bc_fn, right_bc_fn


def put_boundaries(K, r_l):
    """
    Dirichlet boundaries for a vanilla put.
    Hedge requires lending near/below K -> discount at r_l.
    """
    left_bc_fn = lambda tau, S: K * np.exp(-r_l * tau)
    right_bc_fn = lambda tau, S: 0.0
    return left_bc_fn, right_bc_fn


# ==================================================================
# Main solver
# ==================================================================

def bergman_fd_1d(
    S0,
    T,
    r_l,
    r_b,
    sigma,
    payoff_fn,
    K=None,
    left_bc_fn=None,
    right_bc_fn=None,
    M=800,
    N=800,
    S_min_mult=0.02,
    S_max_mult=8.0,
    picard_tol=1e-10,
    picard_maxiter=100,
    rannacher_steps=4,
    return_grid=False,
):
    """
    Finite-difference solver for the 1D Bergman PDE with
    differential lending/borrowing rates.

        V_t + 0.5*sigma^2*S^2*V_SS + r_l*S*V_S
        + (r_b-r_l)*(V-S*V_S)_- - r_l*V = 0

    Parameters
    ----------
    payoff_fn : callable(S: ndarray) -> ndarray
        Terminal payoff V(T, S). Required.
    K : float, optional
        Reference strike, used only to size the grid (S_min/S_max)
        around a sensible region. Pass None if the payoff has no
        natural strike (grid will center on S0 instead).
    left_bc_fn, right_bc_fn : callable(tau: float, S: float) -> float, optional
        Dirichlet boundary values as a function of time-to-maturity
        tau = T - t and the boundary spot level S. Use call_boundaries()
        / put_boundaries() for vanilla payoffs. If omitted, a
        payoff-agnostic linear (zero-gamma) extrapolation boundary
        is used instead.
    """

    # ---------------------------------------------------------
    # 1. Validation
    # ---------------------------------------------------------
    if S0 <= 0:
        raise ValueError("S0 must be strictly positive.")
    if T <= 0:
        raise ValueError("T must be strictly positive.")
    if sigma <= 0:
        raise ValueError("sigma must be strictly positive.")
    if r_b < r_l:
        raise ValueError("Expected r_b >= r_l.")
    if M < 3:
        raise ValueError("M must be at least 3.")
    if N < 1:
        raise ValueError("N must be at least 1.")
    if payoff_fn is None:
        raise ValueError("payoff_fn is required.")

    S0 = float(np.asarray(S0).reshape(-1)[0])
    sigma = float(np.asarray(sigma).reshape(-1)[0])
    if K is not None:
        K = float(np.asarray(K).reshape(-1)[0])

    # ---------------------------------------------------------
    # 2. Spatial grid in x = log(S)
    # ---------------------------------------------------------
    ref = K if K is not None else S0

    S_min = max(S_min_mult * min(S0, ref), 1e-8)
    S_max = S_max_mult * max(S0, ref)

    x_min, x_max = np.log(S_min), np.log(S_max)
    x = np.linspace(x_min, x_max, M + 1, dtype=float)
    h = (x_max - x_min) / M
    S = np.exp(x)
    dt = T / N

    # ---------------------------------------------------------
    # 3. Linear operator coefficients (tridiagonal, interior)
    # ---------------------------------------------------------
    a = 0.5 * sigma**2 / h**2
    conv = r_l - 0.5 * sigma**2
    b_central = conv / (2.0 * h)
    use_upwind = abs(conv) * h > sigma**2

    if not use_upwind:
        lower = np.full(M - 1, a - b_central)
        diag = np.full(M - 1, -2.0 * a - r_l)
        upper = np.full(M - 1, a + b_central)
    else:
        b_up = conv / h
        if conv >= 0:
            lower = np.full(M - 1, a)
            diag = np.full(M - 1, -2.0 * a - b_up - r_l)
            upper = np.full(M - 1, a + b_up)
        else:
            lower = np.full(M - 1, a - b_up)
            diag = np.full(M - 1, -2.0 * a + b_up - r_l)
            upper = np.full(M - 1, a)

    lower = np.asarray(lower, dtype=float).reshape(-1)
    diag = np.asarray(diag, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)

    def apply_L(V):
        V = np.asarray(V, dtype=float).reshape(-1)
        return lower * V[:-2] + diag * V[1:-1] + upper * V[2:]

    def Vx_central(V):
        V = np.asarray(V, dtype=float).reshape(-1)
        return (V[2:] - V[:-2]) / (2.0 * h)

    def reaction(V):
        V = np.asarray(V, dtype=float).reshape(-1)
        Vx = Vx_central(V)
        # (r_b-r_l)*(V - S*V_S)_-  ==  (r_b-r_l)*max(S*V_S - V, 0)
        # S*V_S == V_x in log-space
        return (r_b - r_l) * np.maximum(Vx - V[1:-1], 0.0)

    def solve_tridiag(lo, di, up, rhs):
        n = len(di)
        ab = np.zeros((3, n), dtype=float)
        ab[0, 1:] = up[:-1]
        ab[1, :] = di
        ab[2, :-1] = lo[1:]
        return np.asarray(solve_banded((1, 1), ab, rhs)).reshape(-1)

    # ---------------------------------------------------------
    # 4. Boundary conditions
    # ---------------------------------------------------------
    def boundary_values(tau, V_prev):
        if left_bc_fn is not None:
            left = float(left_bc_fn(tau, S[0]))
        else:
            # Payoff-agnostic default: assume V is locally linear
            # beyond the grid edge (zero-gamma extrapolation),
            # evaluated explicitly from the previous time level.
            left = float(2.0 * V_prev[1] - V_prev[2])

        if right_bc_fn is not None:
            right = float(right_bc_fn(tau, S[-1]))
        else:
            right = float(2.0 * V_prev[-2] - V_prev[-3])

        return left, right

    # ---------------------------------------------------------
    # 5. Terminal condition
    # ---------------------------------------------------------
    V = np.asarray(payoff_fn(S[:, None]), dtype=float).reshape(-1)
    if V.shape != (M + 1,):
        raise ValueError(
            f"payoff_fn must return shape {(M + 1,)}, got {V.shape}"
        )

    # ---------------------------------------------------------
    # 6. Time stepping (Rannacher-smoothed Crank-Nicolson)
    # ---------------------------------------------------------
    for step in range(N):
        tau_old = step * dt
        tau_new = (step + 1) * dt

        if step < rannacher_steps:
            sub_dt = dt / 2.0

            left_bc, right_bc = boundary_values(tau_old + sub_dt, V)
            V = _implicit_step(
                V, sub_dt, apply_L, reaction, lower, diag, upper,
                solve_tridiag, theta=1.0, left_bc=left_bc, right_bc=right_bc,
                picard_tol=picard_tol, picard_maxiter=picard_maxiter,
            )

            left_bc, right_bc = boundary_values(tau_new, V)
            V = _implicit_step(
                V, sub_dt, apply_L, reaction, lower, diag, upper,
                solve_tridiag, theta=1.0, left_bc=left_bc, right_bc=right_bc,
                picard_tol=picard_tol, picard_maxiter=picard_maxiter,
            )
        else:
            left_bc, right_bc = boundary_values(tau_new, V)
            V = _implicit_step(
                V, dt, apply_L, reaction, lower, diag, upper,
                solve_tridiag, theta=0.5, left_bc=left_bc, right_bc=right_bc,
                picard_tol=picard_tol, picard_maxiter=picard_maxiter,
            )

    # ---------------------------------------------------------
    # 7. Price and delta at S0
    # ---------------------------------------------------------
    price = float(np.interp(np.log(S0), x, V))

    dS = np.diff(S)
    dV = np.diff(V)
    S_mid = 0.5 * (S[:-1] + S[1:])
    delta = float(np.interp(S0, S_mid, dV / dS))

    out = {"S": S, "V": V, "x": x} if return_grid else {}
    return price, delta, out


# ==================================================================
# _implicit_step (unchanged)
# ==================================================================

def _implicit_step(
    V, dt, apply_L, reaction, lower, diag, upper,
    solve_tridiag, theta, left_bc, right_bc,
    picard_tol, picard_maxiter,
):
    V = np.asarray(V, dtype=float).reshape(-1)

    L_old = apply_L(V)
    R_old = reaction(V)

    rhs_base = V[1:-1] + (1.0 - theta) * dt * (L_old + R_old)

    lo_imp = -theta * dt * lower
    di_imp = 1.0 - theta * dt * diag
    up_imp = -theta * dt * upper

    V_guess = V.copy()

    for iteration in range(picard_maxiter):
        R_guess = reaction(V_guess)
        rhs = rhs_base + theta * dt * R_guess

        rhs[0] -= lo_imp[0] * left_bc
        rhs[-1] -= up_imp[-1] * right_bc

        V_new_interior = solve_tridiag(lo_imp, di_imp, up_imp, rhs)

        error = np.max(np.abs(V_new_interior - V_guess[1:-1]))
        V_guess[1:-1] = V_new_interior

        if error < picard_tol:
            break
    else:
        raise RuntimeError(
            "Picard iteration did not converge. "
            f"Maximum iterations = {picard_maxiter}, "
            f"last error = {error:.3e}"
        )

    V_guess[0] = left_bc
    V_guess[-1] = right_bc
    return V_guess