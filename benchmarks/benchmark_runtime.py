"""
Wall-clock benchmark: teleporting walkers vs Goodman-Weare.

The paper reports integrated autocorrelation time (IAT) which is the number of steps per
effective sample. That is not the same as time per effective sample. The
teleporting kernel builds an (N, N) weight matrix on every single-walker move
(O(N^2)) while the stretch move is O(N). This measures whether the per-step
cost gap between teleporting and GW reverses the mixing advantage as N grows.

"""

import logging
import statistics
import time

import emcee
import jax
import jax.numpy as jnp
import numpy as np

from teleport.kernels.goodman_weare import goodman_weare_jax, goodman_weare_sequential_jax
from teleport.kernels.teleporting import teleporting_walkers_jax
from teleport.targets import log_prob_doublewell

logging.getLogger("emcee.autocorr").setLevel(logging.ERROR)

TIMING_SWEEPS = 500
IAT_SWEEPS = 5_000
N_REPEATS = 11   
N_SEEDS = 5      
NS = [2, 5, 10, 20, 50]
STEP_SIZE = 0.5
STRETCH_A = 2.0
SEED = 42


def samplers(n, w0, key):
    """{name: (run_fn, subsample)} where run_fn(n_sweeps) returns the chain outputted by the sampler.

    One teleporting step moves a single walker, while one GW step updates all N. The
    teleporting step count is scaled by n so a sweep means N walker-moves for
    all three and its chain is subsampled by n to match the GW recording rate.
    """
    return {
        "teleporting": (
            lambda s: teleporting_walkers_jax(w0, log_prob_doublewell, STEP_SIZE, s * n, key)[1],
            True,
        ),
        "gw_parallel": (
            lambda s: goodman_weare_jax(log_prob_doublewell, w0, s, key, n, STRETCH_A)[0],
            False,
        ),
        "gw_sequential": (
            lambda s: goodman_weare_sequential_jax(log_prob_doublewell, w0, s, key, n, STRETCH_A)[0],
            False,
        ),
    }


def init_walkers(n, key):
    return -0.707 + 0.1 * jax.random.normal(key, shape=(n, 1))


def time_per_sweep(run):
    """(median, min, max) seconds per sweep.

    Warms up once so JIT compilation is excluded, and blocks each timed call so
    JAX's async dispatch completes before the clock stops.
    """
    jax.block_until_ready(run(TIMING_SWEEPS))
    ts = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        jax.block_until_ready(run(TIMING_SWEEPS))
        ts.append(time.perf_counter() - t0)
    return (statistics.median(ts) / TIMING_SWEEPS,
            min(ts) / TIMING_SWEEPS,
            max(ts) / TIMING_SWEEPS)


def iat(chain, n, subsample):
    """(tau in sweeps, reliable). emcee needs ~50*tau samples to trust tau."""
    if subsample:
        chain = chain[::n]
    series = np.asarray(jnp.mean(chain[:, :, 0], axis=1))
    tau = float(emcee.autocorr.integrated_time(series, quiet=True)[0])
    return tau, len(series) >= 50 * tau


def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"x64 enabled: {jax.config.read('jax_enable_x64')}")
    print(f"timing sweeps={TIMING_SWEEPS}, repeats={N_REPEATS}, "
          f"IAT sweeps={IAT_SWEEPS}, seeds={N_SEEDS}, base seed={SEED}\n")

    key = jax.random.PRNGKey(SEED)
    print("| N | sampler | ms/sweep | ms range | IAT median | IAT range | "
          "ms/eff sample | unreliable |")
    print("|---|---|---|---|---|---|---|---|")

    for n in NS:
        key, tkey = jax.random.split(key)
        timings = {
            name: time_per_sweep(run)
            for name, (run, _) in samplers(n, init_walkers(n, tkey), tkey).items()
        }

        taus = {name: [] for name in timings}
        bad = {name: 0 for name in timings}
        for i in range(N_SEEDS):
            key, wkey, rkey = jax.random.split(key, 3)
            w0 = init_walkers(n, wkey)
            for name, (run, sub) in samplers(n, w0, rkey).items():
                tau, ok = iat(run(IAT_SWEEPS), n, sub)
                taus[name].append(tau)
                bad[name] += not ok

        for name, (med, lo, hi) in timings.items():
            t = [x * 1000 for x in (med, lo, hi)]
            tau_med = statistics.median(taus[name])
            flag = "-" if not bad[name] else f"{bad[name]}/{N_SEEDS}"
            note = "" if t[2] / t[1] < 1.5 else "  <- unstable"
            print(f"| {n} | {name} | {t[0]:.4f} | {t[1]:.4f}-{t[2]:.4f} | "
                  f"{tau_med:.1f} | {min(taus[name]):.1f}-{max(taus[name]):.1f} | "
                  f"{t[0] * tau_med:.2f} | {flag} |{note}")


if __name__ == "__main__":
    main()
