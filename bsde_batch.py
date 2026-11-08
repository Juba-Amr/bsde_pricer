
import argparse
import json
import multiprocessing as mp
import os
import time

from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

dims = [2, 3, 5, 10, 20, 50, 100]
seeds = list(range(10))

CONFIG = {
    "K": 100.0,
    "T": 1.0,
    "r": 0.05,
    "sigma": 0.2,
    "rho": 0.3,
    "N": 30,
    "num_epochs": 3000,
    "batch_size": 1024,
    "learning_rate": 1.0,
    "num_units": 10,
    "num_hidden_layers": 3,
    "X0_low": 80.0,
    "X0_high": 120.0,
}

OUTPUT_DIR = "./sweep_checkpoints"
NUM_WORKERS = 8
THREADS_PER_WORKER = 2


# ============================================================
# BUILD MODEL: SAME BSDE AS YOUR NOTEBOOK
# ============================================================

def build_model(d, config=CONFIG, verbose=0):
    import numpy as np
    import tensorflow as tf

    from deep_bsde_model import DeepBSDE
    from sampling import basket_call_payoff

    r = config["r"]
    K = config["K"]

    sigma = np.full(d, config["sigma"], dtype=np.float32)
    weights = np.ones(d, dtype=np.float32) / d

    corr = np.full(
        (d, d), config["rho"], dtype=np.float32
    )
    np.fill_diagonal(corr, 1.0)

    mu_fn = lambda t, x: r * x
    f_fn = lambda t, x, y, z: -r * y
    sigma_fn = lambda t, x: tf.constant(sigma) * x

    payoff_fn = partial(
        basket_call_payoff,
        K=K,
        weights=weights
    )

    model = DeepBSDE(
        d=d,
        N=config["N"],
        X0_low=np.full(d, config["X0_low"], dtype=np.float32),
        X0_high=np.full(d, config["X0_high"], dtype=np.float32),
        payoff_fn=payoff_fn,
        mu_fn=mu_fn,
        sigma_fn=sigma_fn,
        f_fn=f_fn,
        T=config["T"],
        rho=corr,
        num_units=config["num_units"],
        num_hidden_layers=config["num_hidden_layers"],
        verbose=verbose
    )

    return model


# ============================================================
# CHECKPOINT PATHS
# ============================================================

def run_path(root, seed, d):
    return Path(root) / f"seed_{seed:04d}" / f"d_{d:03d}"


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_one(job):
    root, seed, d, config, threads = job

    # Must happen before importing TensorFlow
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    import tensorflow as tf
    import numpy as np

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    # Reproducible seed
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    folder = run_path(root, seed, d)
    weights_file = folder / "model.weights.h5"
    info_file = folder / "info.json"

    # Skip completed models
    if weights_file.exists() and info_file.exists():
        info = json.loads(info_file.read_text())

        if info["config"] == config:
            return seed, d, info["train_seconds"], "skipped"

        raise ValueError(
            f"Existing checkpoint has different configuration: {folder}"
        )

    print(f"Training seed={seed}, d={d}", flush=True)

    model = build_model(d, config)

    t0 = time.perf_counter()

    model.train(
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"]
    )

    train_time = time.perf_counter() - t0

    folder.mkdir(parents=True, exist_ok=True)

    model.save_weights(str(weights_file))

    info = {
        "seed": seed,
        "dimension": d,
        "train_seconds": train_time,
        "config": config
    }

    info_file.write_text(json.dumps(info, indent=2))

    return seed, d, train_time, "trained"


# ============================================================
# PARALLEL SWEEP
# ============================================================

def run_sweep(
    dims=dims,
    seeds=seeds,
    num_workers=NUM_WORKERS,
    threads=THREADS_PER_WORKER,
    output_dir=OUTPUT_DIR,
    config=CONFIG
):

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dimensions": dims,
        "seeds": seeds,
        "config": config
    }

    manifest_file = root / "manifest.json"

    if manifest_file.exists():
        previous = json.loads(manifest_file.read_text())
        if previous != manifest:
            raise ValueError(
                "Existing sweep has different settings. "
                "Use another output directory."
            )
    else:
        manifest_file.write_text(json.dumps(manifest, indent=2))

    jobs = [
        (str(root), seed, d, config, threads)
        for seed in seeds
        for d in dims
    ]

    print(f"Total models: {len(jobs)}")
    print(f"Parallel workers: {num_workers}")

    results = {}

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp.get_context("spawn")
    ) as executor:

        futures = {
            executor.submit(train_one, job): (job[1], job[2])
            for job in jobs
        }

        for future in as_completed(futures):
            seed, d = futures[future]

            try:
                _, _, seconds, status = future.result()

                results[(seed, d)] = seconds

                print(
                    f"seed={seed}, d={d}, "
                    f"{status}, time={seconds:.2f}s",
                    flush=True
                )

            except Exception as e:
                print(f"FAILED seed={seed}, d={d}: {e}", flush=True)
                raise

    return results


# ============================================================
# LOAD ONE TRAINED MODEL
# ============================================================

def load_one(root=OUTPUT_DIR, seed=0, d=2):
    import tensorflow as tf

    folder = run_path(root, seed, d)

    info = json.loads(
        (folder / "info.json").read_text()
    )

    model = build_model(
        d,
        config=info["config"],
        verbose=0
    )

    # Build all neural networks before loading weights
    X0 = tf.ones((1, d), dtype=tf.float32) * info["config"]["X0_low"]

    dW = tf.zeros(
        (1, info["config"]["N"], d),
        dtype=tf.float32
    )

    model(dW, training=False, X0=X0)

    model.load_weights(
        str(folder / "model.weights.h5")
    )

    return model, info["train_seconds"]


# ============================================================
# LOAD ALL MODELS AS LIST OF LISTS
# ============================================================

def load_results(root=OUTPUT_DIR):

    manifest = json.loads(
        (Path(root) / "manifest.json").read_text()
    )

    dimensions = manifest["dimensions"]
    seed_values = manifest["seeds"]

    models = []
    train_times = []

    for seed in seed_values:

        models_seed = []
        times_seed = []

        for d in dimensions:

            model, seconds = load_one(
                root=root,
                seed=seed,
                d=d
            )

            models_seed.append(model)
            times_seed.append(seconds)

        models.append(models_seed)
        train_times.append(times_seed)

    return models, train_times, dimensions, seed_values


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers", type=int, default=NUM_WORKERS
    )
    parser.add_argument(
        "--threads", type=int, default=THREADS_PER_WORKER
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=seeds
    )
    parser.add_argument(
        "--dims", nargs="+", type=int, default=dims
    )
    parser.add_argument(
        "--out", default=OUTPUT_DIR
    )

    args = parser.parse_args()

    run_sweep(
        dims=args.dims,
        seeds=args.seeds,
        num_workers=args.workers,
        threads=args.threads,
        output_dir=args.out
    )
