import tensorflow as tf
import numpy as np

# ==================================================================
# 1D tests — differential-rate Bergman model, S0 scalar
# ==================================================================
tests_1d = {

    # --- Baseline sanity: r_l == r_b collapses to vanilla BS -------
    "1D_NoSpread_ATM_Call": {
        "S0": np.array([100.0]),
        "K": 100.0,
        "T": 1.0,
        "sigma": np.array([0.20]),
        "corr": np.array([[1.0]]),
        "r_l": 0.05,
        "r_b": 0.05,          # spread = 0 -> exact BS closed form applies
        "payoff": lambda S, K=100.0: tf.maximum(S[..., 0] - K, 0.0),
    },

    # --- Standard ITM/ATM/OTM sweep with a real spread -------------
    "1D_ATM_Call": {
        "S0": np.array([100.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20]), "corr": np.array([[1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(S[..., 0] - K, 0.0),
    },
    "1D_ITM_Call": {
        "S0": np.array([120.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20]), "corr": np.array([[1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(S[..., 0] - K, 0.0),
    },
    "1D_OTM_Call": {
        "S0": np.array([80.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20]), "corr": np.array([[1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(S[..., 0] - K, 0.0),
    },
    "1D_Put": {
        "S0": np.array([100.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20]), "corr": np.array([[1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(K - S[..., 0], 0.0),
    },

    # --- Wide rate spread: stresses the nonlinear reaction term ----
    "1D_WideSpread_ATM_Call": {
        "S0": np.array([100.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20]), "corr": np.array([[1.0]]),
        "r_l": 0.00, "r_b": 0.15,   # large r_b - r_l
        "payoff": lambda S, K=100.0: tf.maximum(S[..., 0] - K, 0.0),
    },

    # --- Deep ITM/OTM: intrinsic-value / near-zero regimes ---------
    "1D_DeepITM_Call": {
        "S0": np.array([200.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20]), "corr": np.array([[1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(S[..., 0] - K, 0.0),
    },
    "1D_DeepOTM_Call": {
        "S0": np.array([50.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20]), "corr": np.array([[1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(S[..., 0] - K, 0.0),
    },

    # --- Extreme vol / short maturity: tests grid & step stability -
    "1D_HighVol_ATM_Call": {
        "S0": np.array([100.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.60]), "corr": np.array([[1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(S[..., 0] - K, 0.0),
    },
    "1D_LowVol_ATM_Call": {
        "S0": np.array([100.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.05]), "corr": np.array([[1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(S[..., 0] - K, 0.0),
    },
    "1D_ShortMaturity_ATM_Call": {
        "S0": np.array([100.0]), "K": 100.0, "T": 1.0 / 52,  # 1 week
        "sigma": np.array([0.20]), "corr": np.array([[1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(S[..., 0] - K, 0.0),
    },
    "1D_LongMaturity_ATM_Call": {
        "S0": np.array([100.0]), "K": 100.0, "T": 5.0,
        "sigma": np.array([0.20]), "corr": np.array([[1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(S[..., 0] - K, 0.0),
    },

    # --- Non-monotone payoff: digital / binary call -----------------
    # No closed form under differential rates, but a useful shape
    # check (bounded payoff in [0,1], tests Gibbs-type oscillation).
    "1D_Digital_Call": {
        "S0": np.array([100.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20]), "corr": np.array([[1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.cast(S[..., 0] > K, tf.float32),
    },

    # --- Straddle: tests behavior at a kink from both sides ---------
    "1D_Straddle": {
        "S0": np.array([100.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20]), "corr": np.array([[1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.abs(S[..., 0] - K),
    },
}


# ==================================================================
# ND tests — multi-asset, mostly no closed form under differential
# rates; a few reduce to a known 1D/BS answer as sanity checks.
# ==================================================================
tests_nd = {

    # --- Perfect correlation collapses basket to a single driver ---
    # Known closed form: with corr=1 and equal vol/weight, the
    # basket behaves exactly like a single asset at the same vol.
    "2D_PerfectCorr_Basket": {
        "S0": np.array([100.0, 100.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20, 0.20]),
        "corr": np.array([[1.0, 1.0], [1.0, 1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(
            tf.reduce_mean(S, axis=-1) - K, 0.0
        ),
    },

    # --- Zero rate spread + independent assets: basket vol has a --
    # closed-form (variance of the average), still no exact payoff
    # formula for the American/differential-rate PDE, but useful
    # as a reduced-nonlinearity baseline.
    "2D_NoSpread_IndepBasket": {
        "S0": np.array([100.0, 100.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20, 0.20]),
        "corr": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "r_l": 0.05, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(
            tf.reduce_mean(S, axis=-1) - K, 0.0
        ),
    },

    "2D_Basket_ATM": {
        "S0": np.array([100.0, 100.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20, 0.20]),
        "corr": np.array([[1.0, 0.5], [0.5, 1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(
            tf.reduce_mean(S, axis=-1) - K, 0.0
        ),
    },
    "2D_Basket_LowCorr": {
        "S0": np.array([100.0, 100.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20, 0.20]),
        "corr": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(
            tf.reduce_mean(S, axis=-1) - K, 0.0
        ),
    },
    "2D_Basket_HighCorr": {
        "S0": np.array([100.0, 100.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20, 0.20]),
        "corr": np.array([[1.0, 0.9], [0.9, 1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(
            tf.reduce_mean(S, axis=-1) - K, 0.0
        ),
    },
    "2D_Basket_NegCorr": {
        "S0": np.array([100.0, 100.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20, 0.20]),
        "corr": np.array([[1.0, -0.5], [-0.5, 1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(
            tf.reduce_mean(S, axis=-1) - K, 0.0
        ),
    },

    # --- Spread option: classic no-closed-form case (Kirk's is an
    # approximation, not exact) — good adversarial case -------------
    "2D_Spread_Call": {
        "S0": np.array([110.0, 100.0]), "K": 5.0, "T": 1.0,
        "sigma": np.array([0.25, 0.20]),
        "corr": np.array([[1.0, 0.6], [0.6, 1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=5.0: tf.maximum(S[..., 0] - S[..., 1] - K, 0.0),
    },

    # --- Worst-of put: highly path-/correlation-sensitive ----------
    "2D_WorstOf_Put": {
        "S0": np.array([100.0, 100.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20, 0.30]),
        "corr": np.array([[1.0, 0.3], [0.3, 1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(
            K - tf.reduce_min(S, axis=-1), 0.0
        ),
    },

    "3D_Basket": {
        "S0": np.array([100.0, 100.0, 100.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20, 0.20, 0.20]),
        "corr": np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(
            tf.reduce_mean(S, axis=-1) - K, 0.0
        ),
    },
    "3D_Max_Call": {
        "S0": np.array([100.0, 100.0, 100.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20, 0.20, 0.20]),
        "corr": np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(
            tf.reduce_max(S, axis=-1) - K, 0.0
        ),
    },
    "3D_Min_Call": {
        "S0": np.array([100.0, 100.0, 100.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.20, 0.20, 0.20]),
        "corr": np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(
            tf.reduce_min(S, axis=-1) - K, 0.0
        ),
    },

    # --- Heterogeneous vols/asymmetric spots: no symmetry to lean on
    "3D_Heterogeneous_Basket": {
        "S0": np.array([80.0, 100.0, 130.0]), "K": 100.0, "T": 1.0,
        "sigma": np.array([0.15, 0.25, 0.35]),
        "corr": np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.6], [0.2, 0.6, 1.0]]),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(
            tf.reduce_mean(S, axis=-1) - K, 0.0
        ),
    },

    # --- Curse-of-dimensionality stress test: 5D and 10D max-call --
    # No closed form, but the canonical DeepBSDE benchmark case —
    # good for confirming the network scales where FD grids can't.
    "5D_Max_Call": {
        "S0": np.full(5, 100.0), "K": 100.0, "T": 1.0,
        "sigma": np.full(5, 0.20),
        "corr": 0.3 * np.ones((5, 5)) + 0.7 * np.eye(5),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(
            tf.reduce_max(S, axis=-1) - K, 0.0
        ),
    },
    "10D_Basket": {
        "S0": np.full(10, 100.0), "K": 100.0, "T": 1.0,
        "sigma": np.full(10, 0.20),
        "corr": 0.2 * np.ones((10, 10)) + 0.8 * np.eye(10),
        "r_l": 0.02, "r_b": 0.05,
        "payoff": lambda S, K=100.0: tf.maximum(
            tf.reduce_mean(S, axis=-1) - K, 0.0
        ),
    },
}