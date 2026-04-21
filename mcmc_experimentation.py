#!/usr/bin/env python3
"""
Unified Experiment Pipeline With Dual SH0ES Strategy (Prior-Only and Integrated)
-------------------------------------------------------------------------------
Implements:
- Dual SH0ES modes: prior / forward
- Core cases: LCDM M-marg, w0waCDM M-marg (with/without BAO)
- Optional SN-only degeneracy analysis (H0-M)
- 4 independent MH chains with pilot tuning + production
- Diagnostics: split-Rhat, ESS, acceptance, trace stability
- Publication-style plots
- Optional baseline comparison using existing emcee script as-is

Notes:
- BAO uses anisotropic DESI points (DM/rd, DH/rd) with fixed rd=147.09
- `prior` mode keeps zHD>0.01 preprocessing from current scripts
- `forward` mode uses IS_CALIBRATOR / USED_IN_SH0ES_HF plus CEPH_DIST constraints
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter


# ----------------------------
# Styling
# ----------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "font.size": 10,
    }
)


# ----------------------------
# Constants / baseline data
# ----------------------------
C_LIGHT = 2.998e5  # km/s
RD_FIXED = 147.09  # Mpc, fixed by convention in current scripts

# Same anisotropic BAO set as current baseline script
BAO_DATA = [
    # z      DM/rd   sDM    DH/rd   sDH    rho
    (0.510, 13.62, 0.25, 20.98, 0.61, -0.44),
    (0.706, 16.85, 0.32, 20.08, 0.60, -0.35),
    (0.930, 21.71, 0.28, 17.88, 0.35, -0.38),
    (1.317, 27.79, 0.69, 13.82, 0.42, -0.47),
    (2.330, 39.71, 0.94, 8.52, 0.17, -0.45),
]


# ----------------------------
# Case definitions
# ----------------------------
CASE_LABELS: Dict[str, str] = {
    "lcdm_mmarg": "LCDM M-marg",
    "w0wa_mmarg": "w0waCDM M-marg",
    "lcdm_mfree": "LCDM M-free",
    "w0wa_mfree": "w0waCDM M-free",
}

# Internal case id -> reference model key from mcmc_pantheon+BAO_Prior_Mmarginal.py
CASE_MODEL_MAP: Dict[str, str] = {
    "lcdm_mmarg": "LCDM_Mmarg_NoPrior",
    "w0wa_mmarg": "LCDM_Mmarg_NoPrior_w0wa",
    "lcdm_mfree": "LCDM_Mfree_NoPrior",
    "w0wa_mfree": "LCDM_Mfree_NoPrior_w0wa",
}


@dataclass
class DatasetBundle:
    shoes_mode: str
    z: np.ndarray
    y_obs: np.ndarray
    cov_inv: np.ndarray
    is_calibrator_local: np.ndarray
    calibrator_mu: np.ndarray
    mask_indices: np.ndarray
    info: Dict[str, Any]


# ----------------------------
# Utility
# ----------------------------
def deterministic_seed(base_seed: int, *parts: str) -> int:
    s = str(base_seed) + "::" + "::".join(parts)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def progress_bar(frac: float, width: int = 26) -> str:
    frac = max(0.0, min(1.0, float(frac)))
    n_fill = int(round(frac * width))
    return "█" * n_fill + "·" * (width - n_fill)


def case_plot_path(plot_dir: Path, shoes_mode: str, case_name: str, use_bao: bool, kind: str) -> Path:
    bao_tag = "on" if use_bao else "off"
    return plot_dir / f"{shoes_mode}__{case_name}__bao_{bao_tag}__{kind}.png"


def load_reference_models(models_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load MODELS dictionary directly from the non-emcee baseline file
    (mcmc_pantheon+BAO_Prior_Mmarginal.py) via AST, avoiding import side effects.
    """
    if not models_file.exists():
        raise FileNotFoundError(f"Reference MODELS file not found: {models_file}")

    src = models_file.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(models_file))

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "MODELS":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, dict):
                        raise TypeError("Parsed MODELS object is not a dictionary")
                    return value

    raise KeyError(f"MODELS dictionary not found in {models_file}")


def resolve_reference_model_key(
    case_name: str,
    h0_prior: str,
    reference_models: Dict[str, Dict[str, Any]],
) -> str:
    # Prefer explicit prior-configured models from the reference file when available.
    if case_name == "lcdm_mmarg" and h0_prior == "shoes" and "LCDM_priorSH0ES" in reference_models:
        return "LCDM_priorSH0ES"
    if case_name == "w0wa_mmarg" and h0_prior == "shoes" and "w0waCDM_Prior" in reference_models:
        return "w0waCDM_Prior"

    return CASE_MODEL_MAP[case_name]


def build_case_cfg(
    case_name: str,
    reference_models: Dict[str, Dict[str, Any]],
    h0_prior: str,
) -> Dict[str, Any]:
    if case_name not in CASE_MODEL_MAP:
        raise KeyError(f"Unsupported case name: {case_name}")

    model_key = resolve_reference_model_key(case_name, h0_prior, reference_models)
    if model_key not in reference_models:
        raise KeyError(
            f"Reference model '{model_key}' required by case '{case_name}' "
            "was not found in baseline MODELS."
        )

    m = reference_models[model_key]
    params = list(m["params"])
    fixed = dict(m.get("fixed", {}))
    theta0 = list(m["theta0"])
    step_scales = list(m["step_sizes"])
    prior_bounds = dict(m["prior_bounds"])
    marginalize_M = bool(m.get("marginalize_M", ("M" not in params)))

    return {
        "params": params,
        "fixed": fixed,
        "theta0": theta0,
        "step_scales": step_scales,
        "prior_bounds": prior_bounds,
        "marginalize_M": marginalize_M,
        "label": CASE_LABELS[case_name],
        "source_model": model_key,
        "n_steps_default": int(m.get("n_steps", 4000)),
    }


def load_pantheon_full(data_path: Path, cov_path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(data_path, sep=r"\s+")
    with open(cov_path, "r", encoding="utf-8") as f:
        n_cov = int(f.readline().strip())
        cov_flat = np.array(f.read().split(), dtype=float)
    cov_full = cov_flat.reshape(n_cov, n_cov)
    return df, cov_full


def build_dataset_bundle(df: pd.DataFrame, cov_full: np.ndarray, shoes_mode: str) -> DatasetBundle:
    if shoes_mode == "prior":
        mask = df["zHD"].values > 0.01
        mode_note = "Prior-only baseline: zHD > 0.01 mask"
    elif shoes_mode == "forward":
        mask = (df["IS_CALIBRATOR"].values == 1) | (df["USED_IN_SH0ES_HF"].values == 1)
        mode_note = "Forward integrated: calibrators + SH0ES HF rows"
    else:
        raise ValueError(f"Unsupported shoes mode: {shoes_mode}")

    idx = np.where(mask)[0]
    sub = df.iloc[idx].reset_index(drop=True)
    cov_sub = cov_full[np.ix_(idx, idx)].copy()

    z = sub["zHD"].values.astype(float)
    y_obs = sub["MU_SH0ES"].values.astype(float)

    is_cal = (sub["IS_CALIBRATOR"].values == 1)
    if "CEPH_DIST" not in sub.columns:
        raise KeyError(
            "Forward-mode calibrator constraint requires `CEPH_DIST` column, "
            "but it is missing from Pantheon+SH0ES.dat"
        )
    cal_mu = sub["CEPH_DIST"].values.astype(float)

    if shoes_mode == "forward":
        # Add calibrator host-distance uncertainty on diagonal as requested.
        ceph_err_col = None
        preferred_cols = [
            "CEPH_DIST_ERR",
            "CEPH_DIST_ERROR",
            "CEPH_DIST_ERR_DIAG",
            "CEPH_ERR",
            "SIGMA_CEPH_DIST",
        ]
        for c in preferred_cols:
            if c in sub.columns:
                ceph_err_col = c
                break

        used_fallback = False
        if ceph_err_col is None and "MU_SH0ES_ERR_DIAG" in sub.columns:
            # Fallback for Pantheon+SH0ES table versions that do not carry a
            # dedicated CEPH_DIST_ERR column.
            ceph_err_col = "MU_SH0ES_ERR_DIAG"
            used_fallback = True

        if ceph_err_col is None:
            ceph_err = np.zeros(len(sub), dtype=float)
            used_fallback = True
            ceph_err_col = "NONE(0.0)"
        else:
            ceph_err = sub[ceph_err_col].values.astype(float)

        ceph_err = np.where(np.isfinite(ceph_err) & (ceph_err >= 0.0), ceph_err, 0.0)
        i_cal = np.where(is_cal)[0]
        cov_sub[i_cal, i_cal] += ceph_err[i_cal] ** 2
        mode_note = f"{mode_note} | ceph_err_source={ceph_err_col}"
        if used_fallback:
            print(
                "[forward] warning: dedicated CEPH_DIST_ERR column not found; "
                f"using {ceph_err_col} for calibrator-diagonal inflation.",
                flush=True,
            )

    cov_inv = np.linalg.inv(cov_sub)

    info = {
        "mode_note": mode_note,
        "n_rows": int(len(sub)),
        "n_calibrator": int(np.sum(is_cal)),
        "n_hf": int(np.sum(sub["USED_IN_SH0ES_HF"].values == 1)),
        "z_min": float(np.min(z)),
        "z_max": float(np.max(z)),
    }

    return DatasetBundle(
        shoes_mode=shoes_mode,
        z=z,
        y_obs=y_obs,
        cov_inv=cov_inv,
        is_calibrator_local=is_cal,
        calibrator_mu=cal_mu,
        mask_indices=idx,
        info=info,
    )


# ----------------------------
# Cosmology
# ----------------------------
def E_of_z(z: float, omega_m: float, w0: float, wa: float) -> float:
    omega_de = 1.0 - omega_m
    f_de = (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
    return float(np.sqrt(omega_m * (1 + z) ** 3 + omega_de * f_de))


def luminosity_distance(z: float, h0: float, omega_m: float, w0: float, wa: float) -> float:
    def integrand(zp: float) -> float:
        return 1.0 / E_of_z(zp, omega_m, w0, wa)

    integral, _ = quad(integrand, 0.0, float(z), limit=100)
    return (C_LIGHT / h0) * (1.0 + z) * integral


def distance_modulus_model(z_array: np.ndarray, h0: float, omega_m: float, w0: float, wa: float) -> np.ndarray:
    out = np.zeros_like(z_array, dtype=float)
    for i, z in enumerate(z_array):
        dl = luminosity_distance(float(z), h0, omega_m, w0, wa)
        out[i] = 5 * np.log10(dl) + 25
    return out


def comoving_distance(z: float, h0: float, omega_m: float, w0: float, wa: float) -> float:
    def integrand(zp: float) -> float:
        return 1.0 / E_of_z(zp, omega_m, w0, wa)

    integral, _ = quad(integrand, 0.0, float(z), limit=100)
    return (C_LIGHT / h0) * integral


def DM_over_rd(z: float, h0: float, omega_m: float, w0: float, wa: float, rd: float = RD_FIXED) -> float:
    return comoving_distance(z, h0, omega_m, w0, wa) / rd


def DH_over_rd(z: float, h0: float, omega_m: float, w0: float, wa: float, rd: float = RD_FIXED) -> float:
    return (C_LIGHT / (h0 * E_of_z(z, omega_m, w0, wa))) / rd


# ----------------------------
# Likelihood pieces
# ----------------------------
def build_params(theta: np.ndarray, case_cfg: Dict[str, Any]) -> Dict[str, float]:
    params = dict(zip(case_cfg["params"], theta))
    params.update(case_cfg["fixed"])
    return params


def model_base_vector(dataset: DatasetBundle, mu_cosmo: np.ndarray) -> np.ndarray:
    if dataset.shoes_mode == "forward":
        base = np.where(dataset.is_calibrator_local, dataset.calibrator_mu, mu_cosmo)
    else:
        base = mu_cosmo
    return base


def log_likelihood_sne(theta: np.ndarray, case_cfg: Dict[str, Any], dataset: DatasetBundle) -> float:
    params = build_params(theta, case_cfg)

    h0 = params["H0"]
    omega_m = params["Omega_m"]
    w0 = params["w0"]
    wa = params["wa"]
    M = params["M"]

    if not (0.0 < omega_m < 1.0) or h0 <= 0:
        return -np.inf

    mu_cosmo = distance_modulus_model(dataset.z, h0, omega_m, w0, wa)
    mu_model = model_base_vector(dataset, mu_cosmo) + M

    delta = dataset.y_obs - mu_model
    return float(-0.5 * (delta @ dataset.cov_inv @ delta))


def log_likelihood_sne_marginal(theta: np.ndarray, case_cfg: Dict[str, Any], dataset: DatasetBundle) -> float:
    params = build_params(theta, case_cfg)

    h0 = params["H0"]
    omega_m = params["Omega_m"]
    w0 = params["w0"]
    wa = params["wa"]

    if not (0.0 < omega_m < 1.0) or h0 <= 0:
        return -np.inf

    mu_cosmo = distance_modulus_model(dataset.z, h0, omega_m, w0, wa)
    mu_base = model_base_vector(dataset, mu_cosmo)

    delta = dataset.y_obs - mu_base

    # Analytic marginalization over global offset M (same algebra as current scripts)
    A = float(delta @ dataset.cov_inv @ delta)
    B = float(np.sum(dataset.cov_inv @ delta))
    C = float(np.sum(dataset.cov_inv))

    return float(-0.5 * (A - B**2 / C))


def log_likelihood_bao(theta: np.ndarray, case_cfg: Dict[str, Any]) -> float:
    params = build_params(theta, case_cfg)

    h0 = params["H0"]
    omega_m = params["Omega_m"]
    w0 = params["w0"]
    wa = params["wa"]

    if not (0.0 < omega_m < 1.0) or h0 <= 0:
        return -np.inf

    log_l = 0.0
    for (z, DM_obs, sDM, DH_obs, sDH, rho) in BAO_DATA:
        DM_th = DM_over_rd(z, h0, omega_m, w0, wa, rd=RD_FIXED)
        DH_th = DH_over_rd(z, h0, omega_m, w0, wa, rd=RD_FIXED)

        delta = np.array([DM_obs - DM_th, DH_obs - DH_th], dtype=float)
        cov = np.array(
            [[sDM**2, rho * sDM * sDH], [rho * sDM * sDH, sDH**2]],
            dtype=float,
        )
        cov_inv = np.linalg.inv(cov)
        log_l += float(-0.5 * (delta @ cov_inv @ delta))

    return float(log_l)


def log_prior(theta: np.ndarray, case_cfg: Dict[str, Any], h0_prior: str) -> float:
    params = build_params(theta, case_cfg)
    bounds = case_cfg["prior_bounds"]

    # Apply box bounds only to sampled (free) parameters.
    # Fixed params (e.g. w0/wa in LCDM) may not have entries in prior_bounds.
    for name in case_cfg["params"]:
        value = params[name]
        lo, hi = bounds[name]
        if not (lo < value < hi):
            return -np.inf

    lp = 0.0
    if h0_prior == "shoes":
        lp += -0.5 * ((params["H0"] - 73.04) / 1.04) ** 2
    elif h0_prior == "planck":
        lp += -0.5 * ((params["H0"] - 67.4) / 0.5) ** 2
    elif h0_prior == "none":
        pass
    else:
        raise ValueError(f"Unknown H0 prior option: {h0_prior}")

    return float(lp)


def log_posterior(
    theta: np.ndarray,
    case_cfg: Dict[str, Any],
    dataset: DatasetBundle,
    use_bao: bool,
    h0_prior: str,
) -> float:
    lp = log_prior(theta, case_cfg, h0_prior)
    if not np.isfinite(lp):
        return -np.inf

    if case_cfg["marginalize_M"]:
        ll_sne = log_likelihood_sne_marginal(theta, case_cfg, dataset)
    else:
        ll_sne = log_likelihood_sne(theta, case_cfg, dataset)

    if not np.isfinite(ll_sne):
        return -np.inf

    ll = ll_sne
    if use_bao:
        ll_bao = log_likelihood_bao(theta, case_cfg)
        if not np.isfinite(ll_bao):
            return -np.inf
        ll += ll_bao

    return float(lp + ll)


# ----------------------------
# MH sampler
# ----------------------------
def propose_valid_start(
    rng: np.random.Generator,
    center: np.ndarray,
    scales: np.ndarray,
    case_cfg: Dict[str, Any],
    h0_prior: str,
    dataset: DatasetBundle,
    use_bao: bool,
) -> np.ndarray:
    for _ in range(2000):
        th = center + rng.normal(0.0, scales)
        lp = log_posterior(th, case_cfg, dataset, use_bao, h0_prior)
        if np.isfinite(lp):
            return th
    # deterministic fallback
    return np.array(center, dtype=float)


def run_mh_chain(
    theta0: np.ndarray,
    proposal_cov: np.ndarray,
    n_steps: int,
    seed: int,
    case_cfg: Dict[str, Any],
    dataset: DatasetBundle,
    use_bao: bool,
    h0_prior: str,
    show_progress: bool = False,
    progress_label: str = "",
    progress_updates: int = 20,
    progress_state: Any | None = None,
    progress_key: int = -1,
    progress_lock: Any | None = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    dim = len(theta0)

    chain = np.zeros((n_steps, dim), dtype=float)
    logp = np.zeros(n_steps, dtype=float)

    theta = np.array(theta0, dtype=float)
    lp = log_posterior(theta, case_cfg, dataset, use_bao, h0_prior)

    n_acc = 0
    L = np.linalg.cholesky(proposal_cov)

    t0 = time.perf_counter()
    every = max(1, n_steps // max(1, progress_updates))

    if progress_state is not None and progress_key >= 0:
        if progress_lock is None:
            progress_state[progress_key] = (0, 0.0, float(lp))
        else:
            with progress_lock:
                progress_state[progress_key] = (0, 0.0, float(lp))

    for i in range(n_steps):
        step = L @ rng.normal(size=dim)
        cand = theta + step

        lp_cand = log_posterior(cand, case_cfg, dataset, use_bao, h0_prior)
        if np.isfinite(lp_cand):
            logr = lp_cand - lp
            if np.log(rng.uniform()) < logr:
                theta = cand
                lp = lp_cand
                n_acc += 1

        chain[i] = theta
        logp[i] = lp

        if show_progress and (((i + 1) % every == 0) or (i + 1 == n_steps)):
            frac = (i + 1) / n_steps
            acc_now = n_acc / (i + 1)
            bar = progress_bar(frac)
            msg = (
                f"\r{progress_label} [{bar}] {100*frac:5.1f}% "
                f"step {i+1}/{n_steps}  acc={acc_now:0.3f}  lp={lp: .2f}"
            )
            print(msg, end="", flush=True)

        if progress_state is not None and progress_key >= 0 and (((i + 1) % every == 0) or (i + 1 == n_steps)):
            acc_now = n_acc / (i + 1)
            if progress_lock is None:
                progress_state[progress_key] = (i + 1, float(acc_now), float(lp))
            else:
                with progress_lock:
                    progress_state[progress_key] = (i + 1, float(acc_now), float(lp))

    acc = n_acc / max(1, n_steps)
    elapsed = time.perf_counter() - t0
    if show_progress:
        print("", flush=True)
    return chain, logp, float(acc), float(elapsed)


def run_independent_chains(
    n_chains: int,
    theta_starts: List[np.ndarray],
    proposal_cov: np.ndarray,
    n_steps: int,
    base_seed: int,
    case_name: str,
    shoes_mode: str,
    use_bao: bool,
    stage: str,
    case_cfg: Dict[str, Any],
    dataset: DatasetBundle,
    h0_prior: str,
    parallel: bool,
    parallel_backend: str,
    progress_updates: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    jobs = []
    for i in range(n_chains):
        s = deterministic_seed(base_seed, case_name, shoes_mode, str(use_bao), stage, f"chain{i}")
        jobs.append((theta_starts[i], proposal_cov, n_steps, s, case_cfg, dataset, use_bao, h0_prior))

    chains = [None] * n_chains
    logps = [None] * n_chains
    accs = [None] * n_chains

    if parallel and n_chains > 1:
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED

        if parallel_backend == "thread":
            import threading
            executor_cls = ThreadPoolExecutor
            progress_state = {}
            progress_lock = threading.Lock()
            manager = None
        elif parallel_backend == "process":
            import multiprocessing as mp
            executor_cls = ProcessPoolExecutor
            manager = mp.Manager()
            progress_state = manager.dict()
            progress_lock = manager.Lock()
        else:
            raise ValueError(f"Unsupported parallel backend: {parallel_backend}")

        print(
            f"[{stage}] launching {n_chains} chains in parallel "
            f"(backend={parallel_backend})",
            flush=True,
        )

        with executor_cls(max_workers=n_chains) as ex:
            fut_map = {
                ex.submit(
                    run_mh_chain,
                    *j,
                    False,
                    "",
                    progress_updates,
                    progress_state,
                    i,
                    progress_lock,
                ): i
                for i, j in enumerate(jobs)
            }
            pending = set(fut_map.keys())
            elapseds = [None] * n_chains
            stage_t0 = time.perf_counter()
            last_line_len = 0
            last_render = 0.0
            render_period = 0.25

            while pending:
                done_now, pending = wait(pending, timeout=render_period, return_when=FIRST_COMPLETED)
                for f in done_now:
                    i = fut_map[f]
                    c, lp, a, elapsed = f.result()
                    chains[i] = c
                    logps[i] = lp
                    accs[i] = a
                    elapseds[i] = elapsed

                now = time.perf_counter()
                if (now - last_render) >= render_period or not pending:
                    if progress_lock is None:
                        snap = dict(progress_state)
                    else:
                        with progress_lock:
                            snap = dict(progress_state)

                    steps_now = []
                    acc_now_list = []
                    lp_now_list = []
                    for i in range(n_chains):
                        st, acc_now, lp_now = snap.get(i, (0, 0.0, float("nan")))
                        steps_now.append(float(st))
                        acc_now_list.append(float(acc_now))
                        if np.isfinite(lp_now):
                            lp_now_list.append(float(lp_now))

                    avg_step = float(np.mean(steps_now)) if steps_now else 0.0
                    avg_frac = avg_step / max(1, n_steps)
                    avg_acc = float(np.mean(acc_now_list)) if acc_now_list else 0.0
                    avg_lp = float(np.mean(lp_now_list)) if lp_now_list else float("nan")
                    n_finished = n_chains - len(pending)

                    bar = progress_bar(avg_frac, width=26)
                    lp_txt = f"{avg_lp: .2f}" if np.isfinite(avg_lp) else " nan"
                    line = (
                        f"[{stage}] [{bar}] {100*avg_frac:5.1f}% "
                        f"avg_step={avg_step:6.1f}/{n_steps} avg_acc={avg_acc:0.3f} "
                        f"done={n_finished}/{n_chains} avg_lp={lp_txt}"
                    )
                    pad = " " * max(0, last_line_len - len(line))
                    print("\r" + line + pad, end="", flush=True)
                    last_line_len = len(line)
                    last_render = now

            print("", flush=True)
            mean_acc = float(np.mean(accs))
            mean_elapsed = float(np.mean([x for x in elapseds if x is not None])) if any(x is not None for x in elapseds) else float("nan")
            stage_elapsed = time.perf_counter() - stage_t0
            print(
                f"[{stage}] complete: mean_acc={mean_acc:0.3f}, "
                f"mean_chain_t={mean_elapsed:0.1f}s, wall_t={stage_elapsed:0.1f}s",
                flush=True,
            )

        if manager is not None:
            manager.shutdown()
    else:
        for i, j in enumerate(jobs):
            label = f"[{stage}|chain {i+1}/{n_chains}]"
            c, lp, a, elapsed = run_mh_chain(
                *j,
                True,
                label,
                progress_updates,
            )
            chains[i] = c
            logps[i] = lp
            accs[i] = a
            print(
                f"{label} complete (acc={a:0.3f}, t={elapsed:0.1f}s)",
                flush=True,
            )

    return np.array(chains), np.array(logps), np.array(accs, dtype=float)


def tune_proposal_from_pilot(
    pilot_chains: np.ndarray,
    base_scales: np.ndarray,
) -> np.ndarray:
    # pilot_chains shape: (n_chains, n_steps, n_dim)
    n_ch, n_steps, dim = pilot_chains.shape
    burn = int(0.5 * n_steps)
    samples = pilot_chains[:, burn:, :].reshape(-1, dim)

    if samples.shape[0] < dim + 5:
        return np.diag(base_scales**2)

    emp_cov = np.cov(samples, rowvar=False)
    if dim == 1:
        emp_cov = np.array([[float(emp_cov)]])

    if not np.all(np.isfinite(emp_cov)):
        return np.diag(base_scales**2)

    # Standard random-walk MH scaling rule
    scaled = (2.38**2 / dim) * emp_cov
    jitter = 1e-8 * np.eye(dim)
    cov = scaled + jitter

    # Fallback if near-singular
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = np.diag(base_scales**2)

    return cov


# ----------------------------
# Diagnostics
# ----------------------------
def summarize_posterior(samples: np.ndarray, param_names: List[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for i, p in enumerate(param_names):
        x = samples[:, i]
        q16, q50, q84 = np.percentile(x, [16, 50, 84])
        out[p] = {
            "median": float(q50),
            "lo68": float(q50 - q16),
            "hi68": float(q84 - q50),
            "q16": float(q16),
            "q84": float(q84),
        }
    return out


def split_rhat(chains: np.ndarray) -> Dict[str, float]:
    # chains shape: (m, n, d)
    m, n, d = chains.shape
    n2 = n // 2
    if n2 < 5:
        return {f"param_{i}": float("nan") for i in range(d)}

    split = np.concatenate([chains[:, :n2, :], chains[:, n - n2 :, :]], axis=0)
    m2 = split.shape[0]

    out = {}
    for j in range(d):
        x = split[:, :, j]
        chain_means = x.mean(axis=1)
        chain_vars = x.var(axis=1, ddof=1)
        W = chain_vars.mean()
        B = n2 * chain_means.var(ddof=1)
        var_hat = ((n2 - 1) / n2) * W + B / n2
        rhat = np.sqrt(var_hat / W) if W > 0 else np.nan
        out[f"param_{j}"] = float(rhat)
    return out


def autocorr_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return np.ones(n)
    x = x - np.mean(x)
    var = np.var(x)
    if var == 0:
        return np.ones(n)

    f = np.fft.rfft(np.concatenate([x, np.zeros_like(x)]))
    acf = np.fft.irfft(f * np.conjugate(f))[:n]
    acf /= acf[0]
    return acf


def ess_per_param(chains: np.ndarray, param_names: List[str]) -> Dict[str, float]:
    # Approx ESS via integrated autocorrelation with initial positive sequence
    m, n, d = chains.shape
    out = {}
    for j, name in enumerate(param_names):
        taus = []
        for i in range(m):
            acf = autocorr_1d(chains[i, :, j])
            tau = 1.0
            for k in range(1, len(acf)):
                if acf[k] <= 0:
                    break
                tau += 2.0 * acf[k]
            taus.append(max(tau, 1.0))
        tau_mean = float(np.mean(taus))
        ess = float((m * n) / tau_mean)
        out[name] = ess
    return out


def trace_stability(chains: np.ndarray, param_names: List[str]) -> Dict[str, Dict[str, float]]:
    m, n, d = chains.shape
    out: Dict[str, Dict[str, float]] = {}
    w = max(5, n // 3)

    for j, name in enumerate(param_names):
        drifts = []
        for i in range(m):
            x = chains[i, :, j]
            std = np.std(x, ddof=1)
            if std == 0:
                drifts.append(0.0)
                continue
            first = np.mean(x[:w])
            last = np.mean(x[-w:])
            drifts.append((last - first) / std)
        out[name] = {
            "max_abs_drift_sigma": float(np.max(np.abs(drifts))),
            "mean_abs_drift_sigma": float(np.mean(np.abs(drifts))),
        }
    return out


# ----------------------------
# Plotting
# ----------------------------
def plot_trace_overlay(
    chains: np.ndarray,
    logps: np.ndarray,
    param_names: List[str],
    outpath: Path,
    burn: int,
    title: str,
    pilot_chains: np.ndarray | None = None,
    pilot_logps: np.ndarray | None = None,
) -> None:
    n_ch, n_steps, d = chains.shape
    n_pilot = 0 if pilot_chains is None else pilot_chains.shape[1]
    fig, axes = plt.subplots(d + 1, 1, figsize=(11, 2.2 * (d + 1)), sharex=True)

    colors = plt.cm.tab10(np.linspace(0, 1, n_ch))
    pilot_color = "#7f7f7f"
    for j, p in enumerate(param_names):
        ax = axes[j]
        for i in range(n_ch):
            if pilot_chains is not None:
                x_pilot = np.arange(n_pilot)
                ax.plot(
                    x_pilot,
                    pilot_chains[i, :, j],
                    lw=0.45,
                    alpha=0.6,
                    color=pilot_color,
                    label="pilot" if (j == 0 and i == 0) else None,
                )
            x_prod = np.arange(n_steps) + n_pilot
            ax.plot(
                x_prod,
                chains[i, :, j],
                lw=0.45,
                alpha=0.6,
                color=colors[i],
                label=f"chain {i+1}" if j == 0 else None,
            )
        if n_pilot > 0:
            ax.axvline(n_pilot, ls=":", lw=1.0, color="black", alpha=0.8)
        ax.axvline(n_pilot + burn, ls="--", lw=1.0, color="black", alpha=0.8)
        ax.set_ylabel(p)

    ax = axes[-1]
    for i in range(n_ch):
        if pilot_logps is not None:
            x_pilot = np.arange(n_pilot)
            ax.plot(x_pilot, pilot_logps[i], lw=0.45, alpha=0.6, color=pilot_color)
        x_prod = np.arange(n_steps) + n_pilot
        ax.plot(x_prod, logps[i], lw=0.45, alpha=0.6, color=colors[i])
    if n_pilot > 0:
        ax.axvline(n_pilot, ls=":", lw=1.0, color="black", alpha=0.8)
    ax.axvline(n_pilot + burn, ls="--", lw=1.0, color="black", alpha=0.8)
    ax.set_ylabel("ln post")
    ax.set_xlabel("MCMC step")

    axes[0].legend(ncol=min(n_ch + (1 if n_pilot > 0 else 0), 5), frameon=True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_corner(samples: np.ndarray, param_names: List[str], outpath: Path, title: str) -> None:
    try:
        import corner

        if len(samples) > 40000:
            idx = np.random.default_rng(123).choice(len(samples), 40000, replace=False)
            s = samples[idx]
        else:
            s = samples

        fig = corner.corner(
            s,
            labels=param_names,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3f",
            levels=(0.68, 0.95),
            color="#1f77b4",
            fill_contours=True,
            smooth=1.0,
        )
        fig.suptitle(title, y=1.01)
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        # lightweight fallback
        d = len(param_names)
        fig, axes = plt.subplots(1, d, figsize=(3.5 * d, 3.0))
        if d == 1:
            axes = [axes]
        for i, p in enumerate(param_names):
            axes[i].hist(samples[:, i], bins=60, color="#1f77b4", alpha=0.7)
            axes[i].set_xlabel(p)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(outpath)
        plt.close(fig)


def plot_h0_overlay(
    h0_samples_map: Dict[str, np.ndarray],
    outpath: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, x in h0_samples_map.items():
        hist, bins = np.histogram(x, bins=80, density=True)
        xc = 0.5 * (bins[1:] + bins[:-1])
        ax.plot(xc, hist, lw=2, label=label)
    ax.set_xlabel("H0 [km/s/Mpc]")
    ax.set_ylabel("Posterior density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def contour_levels_from_hist(H: np.ndarray, levels: List[float]) -> List[float]:
    flat = H.flatten()
    idx = np.argsort(flat)[::-1]
    sorted_pdf = flat[idx]
    cdf = np.cumsum(sorted_pdf)
    cdf /= cdf[-1]
    vals = []
    for lv in levels:
        ii = np.searchsorted(cdf, lv)
        ii = min(max(ii, 0), len(sorted_pdf) - 1)
        vals.append(sorted_pdf[ii])
    return vals


def plot_w0_wa_contours(
    s_no_bao: np.ndarray,
    s_bao: np.ndarray,
    idx_w0: int,
    idx_wa: int,
    outpath: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    def draw(samples: np.ndarray, color: str, label: str):
        x = samples[:, idx_w0]
        y = samples[:, idx_wa]
        H, xedges, yedges = np.histogram2d(x, y, bins=80, density=True)
        H = gaussian_filter(H, sigma=1.2)
        lv68, lv95 = contour_levels_from_hist(H, [0.68, 0.95])
        X = 0.5 * (xedges[1:] + xedges[:-1])
        Y = 0.5 * (yedges[1:] + yedges[:-1])
        XX, YY = np.meshgrid(X, Y, indexing="ij")
        cs = ax.contour(XX, YY, H, levels=sorted([lv95, lv68]), colors=[color], linewidths=1.8)
        cs.collections[0].set_label(label)

    draw(s_no_bao, "#1f77b4", "SN only")
    draw(s_bao, "#d62728", "SN + BAO")

    ax.scatter([-1.0], [0.0], marker="*", s=120, c="black", label="(-1, 0)")
    ax.set_xlabel("w0")
    ax.set_ylabel("wa")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_degeneracy_h0_m(
    samples_mfree: np.ndarray,
    idx_h0: int,
    idx_m: int,
    slope: float,
    intercept: float,
    corr: float,
    outpath: Path,
    title: str,
) -> None:
    x = samples_mfree[:, idx_h0]
    y = samples_mfree[:, idx_m]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(x, y, s=3, alpha=0.2, color="#1f77b4")

    xx = np.linspace(np.min(x), np.max(x), 200)
    yy = slope * xx + intercept
    ax.plot(xx, yy, color="#d62728", lw=2.0, label=f"fit slope={slope:.4f}")

    ax.set_xlabel("H0")
    ax.set_ylabel("M")
    ax.set_title(f"{title}\nCorr(H0,M)={corr:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_bao_impact_panel(records: List[Dict[str, Any]], outpath: Path) -> None:
    # Quick visualization of median shifts for H0 and Omega_m between BAO/no-BAO
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    labels, d_h0, d_om = [], [], []
    for r in records:
        labels.append(r["label"])
        d_h0.append(r["delta_H0_median"])
        d_om.append(r["delta_Omega_m_median"])

    y = np.arange(len(labels))
    axes[0].barh(y, d_h0, color="#1f77b4", alpha=0.8)
    axes[0].axvline(0, color="black", lw=1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_title("BAO impact on H0 median")
    axes[0].set_xlabel("ΔH0 (BAO - no BAO)")

    axes[1].barh(y, d_om, color="#d62728", alpha=0.8)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels)
    axes[1].set_title("BAO impact on Ωm median")
    axes[1].set_xlabel("ΔΩm (BAO - no BAO)")

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


# ----------------------------
# Run one experiment
# ----------------------------
def run_case(
    case_name: str,
    shoes_mode: str,
    use_bao: bool,
    case_cfg: Dict[str, Any],
    dataset: DatasetBundle,
    args: argparse.Namespace,
    plot_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dim = len(case_cfg["params"])
    prod_steps = int(args.prod_steps) if args.prod_steps is not None else int(case_cfg["n_steps_default"])

    print(
        f"\n=== RUN case={case_name} ({case_cfg['label']}) "
        f"| shoes={shoes_mode} | bao={int(use_bao)} | chains={args.n_chains} ===",
        flush=True,
    )
    print(
        f"[config] source_model={case_cfg.get('source_model','unknown')} "
        f"theta0={case_cfg['theta0']} step_scales={case_cfg['step_scales']} prod_steps={prod_steps}",
        flush=True,
    )

    center0 = np.array(case_cfg["theta0"], dtype=float)
    base_scales = np.array(case_cfg["step_scales"], dtype=float)
    pilot_chains: np.ndarray | None = None
    pilot_logps: np.ndarray | None = None

    if args.use_pilot and args.pilot_steps > 0:
        print(
            f"[pilot] enabled: {args.pilot_steps} steps/chain, "
            f"{args.n_chains} chains",
            flush=True,
        )
        # Initial dispersed starts for pilot
        rng = np.random.default_rng(deterministic_seed(args.seed, case_name, shoes_mode, str(use_bao), "pilot_init"))
        pilot_starts = [
            propose_valid_start(
                rng=rng,
                center=center0,
                scales=3.0 * base_scales,
                case_cfg=case_cfg,
                h0_prior=args.h0_prior,
                dataset=dataset,
                use_bao=use_bao,
            )
            for _ in range(args.n_chains)
        ]

        pilot_cov0 = np.diag(base_scales**2)

        pilot_chains, pilot_logps, pilot_accs = run_independent_chains(
            n_chains=args.n_chains,
            theta_starts=pilot_starts,
            proposal_cov=pilot_cov0,
            n_steps=args.pilot_steps,
            base_seed=args.seed,
            case_name=case_name,
            shoes_mode=shoes_mode,
            use_bao=use_bao,
            stage="pilot",
            case_cfg=case_cfg,
            dataset=dataset,
            h0_prior=args.h0_prior,
            parallel=args.parallel,
            parallel_backend=args.parallel_backend,
            progress_updates=args.progress_updates,
        )

        tuned_cov = tune_proposal_from_pilot(pilot_chains, base_scales)
        tuned_scales = np.sqrt(np.diag(tuned_cov))
        tuned_scales_str = ", ".join(f"{x:.4g}" for x in tuned_scales)
        print(f"[pilot] tuned proposal std ~ [{tuned_scales_str}]", flush=True)

        pilot_burn = int(0.5 * args.pilot_steps)
        pilot_samples = pilot_chains[:, pilot_burn:, :].reshape(-1, dim)
        pilot_center = np.median(pilot_samples, axis=0)
    else:
        print("[pilot] disabled; using baseline diagonal proposal scales", flush=True)
        pilot_accs = np.array([], dtype=float)
        tuned_cov = np.diag(base_scales**2)
        pilot_center = center0.copy()

    # Production starts around pilot center
    rng2 = np.random.default_rng(deterministic_seed(args.seed, case_name, shoes_mode, str(use_bao), "prod_init"))
    prod_starts = []
    for _ in range(args.n_chains):
        sc = 0.2 * np.sqrt(np.diag(tuned_cov))
        prod_starts.append(
            propose_valid_start(
                rng=rng2,
                center=pilot_center,
                scales=sc,
                case_cfg=case_cfg,
                h0_prior=args.h0_prior,
                dataset=dataset,
                use_bao=use_bao,
            )
        )

    prod_chains, prod_logps, prod_accs = run_independent_chains(
        n_chains=args.n_chains,
        theta_starts=prod_starts,
        proposal_cov=tuned_cov,
        n_steps=prod_steps,
        base_seed=args.seed,
        case_name=case_name,
        shoes_mode=shoes_mode,
        use_bao=use_bao,
        stage="prod",
        case_cfg=case_cfg,
        dataset=dataset,
        h0_prior=args.h0_prior,
        parallel=args.parallel,
        parallel_backend=args.parallel_backend,
        progress_updates=args.progress_updates,
    )

    burn = int(args.burn_frac * prod_steps)
    post = prod_chains[:, burn:, :].reshape(-1, dim)

    # Diagnostics
    rhat_raw = split_rhat(prod_chains)
    rhat_map = {case_cfg["params"][i]: rhat_raw[f"param_{i}"] for i in range(dim)}
    ess_map = ess_per_param(prod_chains, case_cfg["params"])
    stability = trace_stability(prod_chains, case_cfg["params"])

    summary = summarize_posterior(post, case_cfg["params"])
    print(
        f"[prod] acceptance mean={np.mean(prod_accs):0.3f} "
        f"(per-chain: {', '.join(f'{a:0.3f}' for a in prod_accs)})",
        flush=True,
    )
    if args.n_chains > 1:
        rhat_items = ", ".join(
            f"{p}={rhat_map[p]:0.4f}" if np.isfinite(rhat_map[p]) else f"{p}=nan"
            for p in case_cfg["params"]
        )
        finite_rhats = [rhat_map[p] for p in case_cfg["params"] if np.isfinite(rhat_map[p])]
        max_rhat = max(finite_rhats) if finite_rhats else float("nan")
        print(f"[diag] split-Rhat: {rhat_items}", flush=True)
        print(f"[diag] max split-Rhat: {max_rhat:0.4f}", flush=True)
    h0_median = summary.get("H0", {}).get("median", float("nan"))
    om_median = summary.get("Omega_m", {}).get("median", float("nan"))
    print(f"[summary] H0={h0_median:.4f}, Omega_m={om_median:.4f}", flush=True)

    # Plots
    run_title = f"{case_cfg['label']} | shoes={shoes_mode} | bao={use_bao}"
    plot_trace_overlay(
        prod_chains,
        prod_logps,
        case_cfg["params"],
        case_plot_path(plot_dir, shoes_mode, case_name, use_bao, "trace_overlay"),
        burn=burn,
        title=run_title,
        pilot_chains=pilot_chains,
        pilot_logps=pilot_logps,
    )
    plot_corner(post, case_cfg["params"], case_plot_path(plot_dir, shoes_mode, case_name, use_bao, "corner"), run_title)

    result = {
        "case": case_name,
        "case_label": case_cfg["label"],
        "source_model": case_cfg.get("source_model"),
        "shoes_mode": shoes_mode,
        "use_bao": bool(use_bao),
        "h0_prior": args.h0_prior,
        "n_chains": int(args.n_chains),
        "parallel_backend": args.parallel_backend,
        "pilot_steps": int(args.pilot_steps),
        "use_pilot": bool(args.use_pilot),
        "prod_steps": int(prod_steps),
        "burn_steps": int(burn),
        "acceptance": {
            "pilot_mean": float(np.mean(pilot_accs)) if len(pilot_accs) > 0 else None,
            "pilot_per_chain": [float(x) for x in pilot_accs],
            "prod_mean": float(np.mean(prod_accs)),
            "prod_per_chain": [float(x) for x in prod_accs],
        },
        "diagnostics": {
            "rhat": rhat_map,
            "ess": ess_map,
            "trace_stability": stability,
        },
        "posterior_summary": summary,
        "param_names": case_cfg["params"],
    }

    payload = {
        "samples": post,
        "chains": prod_chains,
        "logps": prod_logps,
        "param_names": case_cfg["params"],
        "summary": summary,
    }

    return result, payload


# ----------------------------
# Emcee baseline comparison
# ----------------------------
def load_emcee_module(path: Path):
    spec = importlib.util.spec_from_file_location("baseline_emcee_module", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_emcee_baseline_compare(
    repo_dir: Path,
    mh_result_index: Dict[Tuple[str, str, bool], Dict[str, Any]],
) -> Dict[str, Any]:
    emcee_file = repo_dir / "mcmc_pantheon+BAO_Prior_Mmarginal_emcee.py"
    mod = load_emcee_module(emcee_file)

    z, mu_obs, cov_inv = mod.load_data()

    mapping = {
        "lcdm_mmarg": "LCDM_Mmarg_NoPrior",
        "w0wa_mmarg": "LCDM_Mmarg_NoPrior_w0wa",
    }

    comparisons = {}
    for case_name, model_name in mapping.items():
        cfg = mod.MODELS[model_name]
        chain, logp = mod.run_mcmc_emcee(z, mu_obs, cov_inv, model_cfg=cfg, n_steps=5000, n_walkers=32)

        burn = int(0.3 * len(chain))
        post = chain[burn:]

        # summarize emcee
        emcee_summary = {}
        for i, p in enumerate(cfg["params"]):
            q16, q50, q84 = np.percentile(post[:, i], [16, 50, 84])
            emcee_summary[p] = {
                "median": float(q50),
                "lo68": float(q50 - q16),
                "hi68": float(q84 - q50),
            }

        key = ("prior", case_name, True)
        mh = mh_result_index.get(key)

        delta = {}
        if mh is not None:
            for p in cfg["params"]:
                if p in mh["posterior_summary"]:
                    delta[p] = {
                        "delta_median_mh_minus_emcee": float(
                            mh["posterior_summary"][p]["median"] - emcee_summary[p]["median"]
                        ),
                        "mh_lo68": float(mh["posterior_summary"][p]["lo68"]),
                        "mh_hi68": float(mh["posterior_summary"][p]["hi68"]),
                        "emcee_lo68": float(emcee_summary[p]["lo68"]),
                        "emcee_hi68": float(emcee_summary[p]["hi68"]),
                    }

        comparisons[case_name] = {
            "emcee_model": model_name,
            "emcee_summary": emcee_summary,
            "delta_vs_mh": delta,
        }

    return comparisons


# ----------------------------
# Main orchestration
# ----------------------------
def build_run_list(args: argparse.Namespace) -> List[Tuple[str, bool]]:
    if args.case == "core4":
        return [
            ("lcdm_mmarg", False),
            ("lcdm_mmarg", True),
            ("w0wa_mmarg", False),
            ("w0wa_mmarg", True),
        ]

    if args.use_bao is None:
        return [(args.case, False), (args.case, True)]

    return [(args.case, bool(args.use_bao))]


def run_degeneracy_block(
    shoes_mode: str,
    dataset: DatasetBundle,
    reference_models: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
    plot_dir: Path,
    result_index: Dict[Tuple[str, str, bool], Dict[str, Any]],
    payload_index: Dict[Tuple[str, str, bool], Dict[str, Any]],
) -> Dict[str, Any]:
    # Run M-free SN-only LCDM to quantify H0-M degeneracy
    deg_case = "lcdm_mfree"
    deg_cfg = build_case_cfg(deg_case, reference_models, args.h0_prior)
    use_bao = False

    deg_result, deg_payload = run_case(deg_case, shoes_mode, use_bao, deg_cfg, dataset, args, plot_dir)

    key_mmarg = (shoes_mode, "lcdm_mmarg", False)
    if key_mmarg in result_index:
        mmarg_result = result_index[key_mmarg]
        mmarg_payload = payload_index[key_mmarg]
    else:
        # If not already run, run supporting mmarg no-BAO
        mmarg_cfg = build_case_cfg("lcdm_mmarg", reference_models, args.h0_prior)
        mmarg_result, mmarg_payload = run_case("lcdm_mmarg", shoes_mode, False, mmarg_cfg, dataset, args, plot_dir)
        result_index[key_mmarg] = mmarg_result
        payload_index[key_mmarg] = mmarg_payload

    # Quantify H0-M direction
    pnames = deg_payload["param_names"]
    idx_h0 = pnames.index("H0")
    idx_m = pnames.index("M")
    s = deg_payload["samples"]
    x = s[:, idx_h0]
    y = s[:, idx_m]
    slope, intercept = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1])

    plot_degeneracy_h0_m(
        samples_mfree=s,
        idx_h0=idx_h0,
        idx_m=idx_m,
        slope=float(slope),
        intercept=float(intercept),
        corr=corr,
        outpath=plot_dir / f"{shoes_mode}__lcdm_mfree__bao_off__h0_m_degeneracy.png",
        title=f"H0-M Degeneracy ({shoes_mode} mode)",
    )

    # Geometry/mixing change after M marginalization
    h0_ess_mfree = deg_result["diagnostics"]["ess"]["H0"]
    h0_ess_mmarg = mmarg_result["diagnostics"]["ess"]["H0"]
    acc_mfree = deg_result["acceptance"]["prod_mean"]
    acc_mmarg = mmarg_result["acceptance"]["prod_mean"]

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].bar(["M-free", "M-marg"], [h0_ess_mfree, h0_ess_mmarg], color=["#1f77b4", "#d62728"])
    ax[0].set_title("ESS(H0) comparison")
    ax[0].set_ylabel("ESS")

    ax[1].bar(["M-free", "M-marg"], [acc_mfree, acc_mmarg], color=["#1f77b4", "#d62728"])
    ax[1].set_title("Acceptance comparison")
    ax[1].set_ylabel("Acceptance")

    fig.suptitle(f"SN-only geometry/mixing change ({shoes_mode})")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{shoes_mode}__lcdm_mfree__bao_off__mixing_comparison.png")
    plt.close(fig)

    out = {
        "shoes_mode": shoes_mode,
        "slope_M_vs_H0": float(slope),
        "intercept_M_vs_H0": float(intercept),
        "corr_H0_M": corr,
        "H0_ESS_mfree": float(h0_ess_mfree),
        "H0_ESS_mmarg": float(h0_ess_mmarg),
        "acceptance_mfree": float(acc_mfree),
        "acceptance_mmarg": float(acc_mmarg),
    }

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified MH experimentation pipeline with dual SH0ES strategy.\n"
            "Runs Pantheon(+BAO) cases with configurable SH0ES treatment, diagnostics, and plots."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 mcmc_experimentation.py --case core4 --shoes-mode both\n"
            "  python3 mcmc_experimentation.py --case lcdm_mmarg --shoes-mode prior --use-bao --h0-prior shoes\n"
            "  python3 mcmc_experimentation.py --case lcdm_mmarg --shoes-mode forward --no-use-bao --n-chains 4\n"
            "  python3 mcmc_experimentation.py --compare-emcee --parallel-backend process\n"
        ),
    )
    parser.add_argument(
        "--case",
        choices=["core4", "lcdm_mmarg", "w0wa_mmarg"],
        default="core4",
        help=(
            "Case selector.\n"
            "  core4: run the 4-case matrix (lcdm_mmarg/w0wa_mmarg, each with and without BAO)\n"
            "  lcdm_mmarg: run only LCDM with M analytically marginalized\n"
            "  w0wa_mmarg: run only w0waCDM with M analytically marginalized"
        ),
    )
    parser.add_argument(
        "--shoes-mode",
        choices=["prior", "forward", "both"],
        default="both",
        help=(
            "SH0ES handling strategy.\n"
            "  prior: current-style cosmology sample (zHD > 0.01)\n"
            "  forward: integrated calibrator/HF likelihood using CEPH_DIST constraints\n"
            "  both: run prior and forward modes"
        ),
    )
    parser.add_argument(
        "--use-bao",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Toggle BAO block for single-case runs.\n"
            "  --use-bao enables BAO\n"
            "  --no-use-bao disables BAO\n"
            "If omitted, single-case runs execute both BAO off and BAO on."
        ),
    )
    parser.add_argument(
        "--n-chains",
        type=int,
        default=4,
        help="Number of independent MH chains per run (default: 4).",
    )
    parser.add_argument(
        "--use-pilot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable/disable pilot stage used to tune proposal covariance.\n"
            "  --use-pilot (default)\n"
            "  --no-use-pilot"
        ),
    )
    parser.add_argument(
        "--pilot-steps",
        type=int,
        default=1200,
        help="Pilot MH steps per chain when pilot is enabled (default: 1200).",
    )
    parser.add_argument(
        "--prod-steps",
        type=int,
        default=None,
        help=(
            "Production MH steps per chain.\n"
            "If omitted, uses `n_steps` from the mapped reference MODELS entry."
        ),
    )
    parser.add_argument(
        "--progress-updates",
        type=int,
        default=20,
        help=(
            "Approximate number of progress updates per chain stage.\n"
            "Used for compact progress bars/completion messages (default: 20)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed for deterministic per-chain/per-case seeds (default: 42).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="experiments_unified",
        help=(
            "Deprecated/ignored compatibility option.\n"
            "Structured non-plot artifacts are disabled; only --plot-dir is used."
        ),
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="experimentation_plots",
        help="Directory where all plot PNGs are written with case-based names (default: experimentation_plots).",
    )
    parser.add_argument(
        "--compare-emcee",
        action="store_true",
        help="Run baseline emcee comparison block using existing emcee script/settings as-is.",
    )
    parser.add_argument(
        "--h0-prior",
        choices=["none", "shoes", "planck"],
        default="none",
        help=(
            "Optional Gaussian prior on H0.\n"
            "  none: no Gaussian H0 prior (default)\n"
            "  shoes: N(73.04, 1.04)\n"
            "  planck: N(67.4, 0.5)"
        ),
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable/disable parallel chain execution.\n"
            "  --parallel (default)\n"
            "  --no-parallel"
        ),
    )
    parser.add_argument(
        "--parallel-backend",
        choices=["thread", "process"],
        default="thread",
        help=(
            "Backend for parallel chains (used when --parallel and n-chains>1).\n"
            "  thread: thread workers (default)\n"
            "  process: process workers (usually better CPU scaling)"
        ),
    )
    parser.add_argument(
        "--burn-frac",
        type=float,
        default=0.3,
        help="Fraction of production samples discarded as burn-in (default: 0.3).",
    )
    parser.add_argument(
        "--run-degeneracy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable/disable SN-only H0-M degeneracy diagnostics block.\n"
            "  --run-degeneracy (default)\n"
            "  --no-run-degeneracy"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_dir = Path.cwd()
    models_file = repo_dir / "mcmc_pantheon+BAO_Prior_Mmarginal.py"
    reference_models = load_reference_models(models_file)
    plot_dir = Path(args.plot_dir).resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)

    data_path = repo_dir / "Pantheon+SH0ES.dat"
    cov_path = repo_dir / "Pantheon+SH0ES_STAT+SYS.cov"

    if not data_path.exists() or not cov_path.exists():
        raise FileNotFoundError("Pantheon+SH0ES data/covariance files not found in current directory")

    df, cov_full = load_pantheon_full(data_path, cov_path)
    # Build datasets for requested SH0ES modes
    if args.shoes_mode == "both":
        shoes_modes = ["prior", "forward"]
    else:
        shoes_modes = [args.shoes_mode]

    datasets = {mode: build_dataset_bundle(df, cov_full, mode) for mode in shoes_modes}

    run_list = build_run_list(args)

    run_results: List[Dict[str, Any]] = []
    result_index: Dict[Tuple[str, str, bool], Dict[str, Any]] = {}
    payload_index: Dict[Tuple[str, str, bool], Dict[str, Any]] = {}
    total_primary_runs = len(shoes_modes) * len(run_list)
    completed_primary_runs = 0
    t_all0 = time.perf_counter()
    print(
        f"Starting experimentation: {total_primary_runs} primary run(s), "
        f"shoes_mode={args.shoes_mode}, case={args.case}",
        flush=True,
    )

    for mode in shoes_modes:
        dataset = datasets[mode]

        for case_name, use_bao in run_list:
            case_cfg = build_case_cfg(case_name, reference_models, args.h0_prior)
            completed_primary_runs += 1
            print(
                f"\n[run {completed_primary_runs}/{total_primary_runs}] "
                f"mode={mode}, case={case_name}, bao={int(use_bao)}",
                flush=True,
            )
            t0 = time.perf_counter()
            result, payload = run_case(case_name, mode, use_bao, case_cfg, dataset, args, plot_dir)
            dt = time.perf_counter() - t0
            print(
                f"[run {completed_primary_runs}/{total_primary_runs}] done in {dt:0.1f}s",
                flush=True,
            )

            run_results.append(result)
            key = (mode, case_name, use_bao)
            result_index[key] = result
            payload_index[key] = payload

        if args.run_degeneracy:
            run_degeneracy_block(
                shoes_mode=mode,
                dataset=dataset,
                reference_models=reference_models,
                args=args,
                plot_dir=plot_dir,
                result_index=result_index,
                payload_index=payload_index,
            )

    # Global posterior overlays and BAO comparison panels
    # 1) H0 overlays per (mode, case) BAO on/off
    bao_impact_records: List[Dict[str, Any]] = []
    for mode in shoes_modes:
        for case_name in ["lcdm_mmarg", "w0wa_mmarg"]:
            k_no = (mode, case_name, False)
            k_yes = (mode, case_name, True)
            if k_no in payload_index and k_yes in payload_index:
                p_no = payload_index[k_no]
                p_yes = payload_index[k_yes]

                idx_h0_no = p_no["param_names"].index("H0")
                idx_h0_yes = p_yes["param_names"].index("H0")
                idx_om_no = p_no["param_names"].index("Omega_m")
                idx_om_yes = p_yes["param_names"].index("Omega_m")

                h0_no = p_no["samples"][:, idx_h0_no]
                h0_yes = p_yes["samples"][:, idx_h0_yes]

                plot_h0_overlay(
                    {"SN only": h0_no, "SN + BAO": h0_yes},
                    plot_dir / f"{mode}__{case_name}__bao_compare__h0_overlay.png",
                    title=f"H0 posterior overlay ({mode}, {case_name})",
                )

                # BAO impact record
                med_h0_no = result_index[k_no]["posterior_summary"]["H0"]["median"]
                med_h0_yes = result_index[k_yes]["posterior_summary"]["H0"]["median"]
                med_om_no = result_index[k_no]["posterior_summary"]["Omega_m"]["median"]
                med_om_yes = result_index[k_yes]["posterior_summary"]["Omega_m"]["median"]

                bao_impact_records.append(
                    {
                        "label": f"{mode}:{case_name}",
                        "shoes_mode": mode,
                        "case": case_name,
                        "delta_H0_median": float(med_h0_yes - med_h0_no),
                        "delta_Omega_m_median": float(med_om_yes - med_om_no),
                    }
                )

                # w0-wa contours for w0wa case
                if case_name == "w0wa_mmarg":
                    idx_w0_no = p_no["param_names"].index("w0")
                    idx_wa_no = p_no["param_names"].index("wa")
                    idx_w0_yes = p_yes["param_names"].index("w0")
                    idx_wa_yes = p_yes["param_names"].index("wa")

                    # reorder to use same columns from each sample object
                    s_no = np.column_stack([p_no["samples"][:, idx_w0_no], p_no["samples"][:, idx_wa_no]])
                    s_yes = np.column_stack([p_yes["samples"][:, idx_w0_yes], p_yes["samples"][:, idx_wa_yes]])

                    plot_w0_wa_contours(
                        s_no_bao=s_no,
                        s_bao=s_yes,
                        idx_w0=0,
                        idx_wa=1,
                        outpath=plot_dir / f"{mode}__{case_name}__bao_compare__w0_wa_contours.png",
                        title=f"w0-wa contours ({mode})",
                    )

    if bao_impact_records:
        plot_bao_impact_panel(bao_impact_records, plot_dir / "global__bao_impact_panel.png")

    # global H0 overlay across all runs
    h0_overlay_map = {}
    for r in run_results:
        k = (r["shoes_mode"], r["case"], r["use_bao"])
        payload = payload_index.get(k)
        if payload is None:
            continue
        idx_h0 = payload["param_names"].index("H0")
        lbl = f"{r['shoes_mode']} | {r['case']} | bao={int(r['use_bao'])}"
        h0_overlay_map[lbl] = payload["samples"][:, idx_h0]

    if h0_overlay_map:
        plot_h0_overlay(h0_overlay_map, plot_dir / "global__h0_overlays_all_runs.png", "H0 posterior overlays across runs")

    # Optional emcee comparison (baseline only, no emcee modifications)
    emcee_comparison = None
    if args.compare_emcee:
        emcee_comparison = run_emcee_baseline_compare(
            repo_dir=repo_dir,
            mh_result_index=result_index,
        )

    print("\nDone.")
    print(f"Plots written to:   {plot_dir}")
    print(f"Total elapsed:      {time.perf_counter() - t_all0:0.1f}s")


if __name__ == "__main__":
    main()
