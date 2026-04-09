"""
MCMC con emcee — Pantheon+ Supernovae
======================================
Usa l'affine-invariant ensemble sampler (Goodman & Weare 2010).
Parametri: H0, Omega_m, w0, wa  (M marginalizzato analiticamente)

Installazione:
  pip install numpy scipy pandas matplotlib emcee corner tqdm requests
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pandas as pd
import emcee
import corner
import os, requests

# ─────────────────────────────────────────────
# 1. DOWNLOAD E CARICAMENTO
# ─────────────────────────────────────────────

def download_pantheon(data_path="Pantheon+SH0ES.dat",
                      cov_path="Pantheon+SH0ES_STAT+SYS.cov"):
    base = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/"
            "DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/")
    if not os.path.exists(data_path):
        print("Scaricando dati...")
        r = requests.get(base + "Pantheon%2BSH0ES.dat")
        open(data_path, "w").write(r.text)
    if not os.path.exists(cov_path):
        print("Scaricando covarianza...")
        r = requests.get(base + "Pantheon%2BSH0ES_STAT%2BSYS.cov")
        open(cov_path, "w").write(r.text)

def load_data(data_path="Pantheon+SH0ES.dat",
              cov_path="Pantheon+SH0ES_STAT+SYS.cov"):
    df_full = pd.read_csv(data_path, sep=r'\s+')
    mask    = df_full["zHD"] > 0.01
    df      = df_full[mask].reset_index(drop=True)

    z      = df["zHD"].values
    mu_obs = df["MU_SH0ES"].values
    n      = len(z)
    print(f"Supernovae caricate: {n}")

    with open(cov_path) as f:
        n_cov    = int(f.readline())
        cov_flat = np.array(f.read().split(), dtype=float)
    cov_full = cov_flat.reshape(n_cov, n_cov)

    idx     = np.where(mask.values)[0]
    cov     = cov_full[np.ix_(idx, idx)]
    cov_inv = np.linalg.inv(cov)
    print("Matrice di covarianza invertita.")
    return z, mu_obs, cov_inv


# ─────────────────────────────────────────────
# 2. MODELLO COSMOLOGICO
# ─────────────────────────────────────────────

c_light = 2.998e5  # km/s

def E(z, Omega_m, w0, wa):
    """H(z)/H0 per il modello w0waCDM con universo piatto."""
    Omega_de = 1.0 - Omega_m
    f_de     = (1 + z)**(3*(1 + w0 + wa)) * np.exp(-3*wa*z/(1+z))
    return np.sqrt(Omega_m*(1+z)**3 + Omega_de*f_de)

def mu_theory(z_arr, H0, Omega_m, w0, wa):
    """
    Modulo di distanza teorico per un array di redshift.
    mu = 5*log10(dL/10pc) = 5*log10(dL [Mpc]) + 25
    """
    mu = np.empty(len(z_arr))
    for i, z in enumerate(z_arr):
        integral, _ = quad(lambda zp: 1.0/E(zp, Omega_m, w0, wa),
                           0, z, limit=100)
        dL   = (c_light/H0) * (1+z) * integral  # Mpc
        mu[i] = 5*np.log10(dL) + 25
    return mu


# ─────────────────────────────────────────────
# 3. LIKELIHOOD (M marginalizzato analiticamente)
# ─────────────────────────────────────────────

def log_likelihood(theta, z, mu_obs, cov_inv):
    """
    Likelihood con M marginalizzato analiticamente.

    Siccome M entra linearmente: mu_model = mu_cosmo + M
    il valore ottimale di M è:
        M* = (1^T C^{-1} delta) / (1^T C^{-1} 1)
    e la likelihood marginalizzata è:
        ln L = -1/2 * (delta^T C^{-1} delta - B^2/C_val)
    dove delta = mu_obs - mu_cosmo (senza M).
    """
    H0, Omega_m, w0, wa = theta

    mu_th = mu_theory(z, H0, Omega_m, w0, wa)
    delta = mu_obs - mu_th

    ones  = np.ones(len(z))
    A     = delta @ cov_inv @ delta
    B     = delta @ cov_inv @ ones
    C_val = ones  @ cov_inv @ ones

    return -0.5 * (A - B**2 / C_val)

def log_prior(theta):
    """Prior piatte entro range fisici."""
    H0, Omega_m, w0, wa = theta
    if (50  < H0      < 90  and
        0.1 < Omega_m < 0.6 and
       -2.0 < w0      < 0.0 and
       -3.0 < wa      < 3.0):
        return 0.0
    return -np.inf

def log_posterior(theta, z, mu_obs, cov_inv):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z, mu_obs, cov_inv)


# ─────────────────────────────────────────────
# 4. SETUP EMCEE
# ─────────────────────────────────────────────

def run_emcee(z, mu_obs, cov_inv,
              n_walkers=32,
              n_steps=3000,
              n_burnin=500):
    """
    Esegue l'MCMC con emcee.

    n_walkers : numero di walker paralleli (deve essere pari e >= 2*n_params)
    n_steps   : passi per walker DOPO il burn-in
    n_burnin  : passi di burn-in da scartare
    """
    n_params = 4
    # [H0,   Omega_m, w0,   wa  ]
    theta0   = np.array([70.0, 0.31, -1.0, 0.0])

    # Inizializza i walker in una piccola sfera attorno a theta0
    # La sfera deve essere piccola ma non zero — se tutti i walker
    # partono dallo stesso punto l'ensemble non funziona
    rng = np.random.default_rng(42)
    pos = theta0 + rng.normal(0, 1e-3, size=(n_walkers, n_params))
    pos[:, 0] *= 1.0   # H0: scala di ~1 km/s/Mpc
    pos[:, 1] *= 0.01  # Omega_m: scala di ~0.01
    pos[:, 2] *= 0.05  # w0
    pos[:, 3] *= 0.1   # wa

    # Ricrea pos con scale appropriate
    scales = np.array([1.0, 0.01, 0.05, 0.1])
    pos    = theta0 + rng.normal(0, scales, size=(n_walkers, n_params))

    sampler = emcee.EnsembleSampler(
        n_walkers, n_params, log_posterior,
        args=(z, mu_obs, cov_inv)
    )

    # Fase 1: burn-in (scartata)
    print(f"\nBurn-in: {n_burnin} passi × {n_walkers} walker...")
    pos, _, _ = sampler.run_mcmc(pos, n_burnin, progress=True)
    sampler.reset()  # svuota la memoria, riparte da pos attuale

    # Fase 2: catena vera
    print(f"\nCatena principale: {n_steps} passi × {n_walkers} walker...")
    sampler.run_mcmc(pos, n_steps, progress=True)

    # Statistiche di convergenza
    try:
        tau = sampler.get_autocorr_time()
        print(f"\nTempo di autocorrelazione stimato:")
        for name, t in zip(["H0", "Omega_m", "w0", "wa"], tau):
            print(f"  {name:8s}: {t:.1f} passi")
        print(f"  → catena efficace: ~{n_steps * n_walkers / np.mean(tau):.0f} campioni indipendenti")
    except emcee.autocorr.AutocorrError:
        print("\nNon abbastanza passi per stimare l'autocorrelazione — aumenta n_steps.")

    return sampler


# ─────────────────────────────────────────────
# 5. ANALISI E PLOT
# ─────────────────────────────────────────────

def analyze_and_plot(sampler, z, mu_obs):
    param_names  = ["H0", "Omega_m", "w0", "wa"]
    param_labels = [r"$H_0$", r"$\Omega_m$", r"$w_0$", r"$w_a$"]

    # Estrai catena flat (tutti i walker appiattiti in una lista)
    # thin=1 per ora — puoi aumentarlo se l'autocorrelazione è alta
    flat_chain = sampler.get_chain(flat=True)
    print(f"\nCampioni totali nella catena flat: {len(flat_chain)}")

    # ── Risultati numerici ──
    print("\nRisultati (mediana e intervallo 68%):")
    print("-" * 45)
    for i, name in enumerate(param_names):
        samples = flat_chain[:, i]
        med = np.median(samples)
        lo  = np.percentile(samples, 16)
        hi  = np.percentile(samples, 84)
        print(f"  {name:10s} = {med:.4f}  +{hi-med:.4f} / -{med-lo:.4f}")

    # ── Trace plot ──
    # Qui vedi tutti i walker sovrapposti per ogni parametro
    fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 8), sharex=True)
    chain_full = sampler.get_chain()  # shape: (n_steps, n_walkers, n_params)
    for i, (ax, label) in enumerate(zip(axes, param_labels)):
        for walker in range(chain_full.shape[1]):
            ax.plot(chain_full[:, walker, i],
                    color="steelblue", alpha=0.15, lw=0.5)
        ax.set_ylabel(label, fontsize=11)
    axes[-1].set_xlabel("passo", fontsize=11)
    fig.suptitle("Trace plots — tutti i walker", fontsize=13)
    plt.tight_layout()
    plt.savefig("trace_emcee.png", dpi=150)
    plt.show()
    print("Salvato: trace_emcee.png")

    # ── Corner plot ──
    fig = corner.corner(
        flat_chain,
        labels=param_labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 11},
        smooth=1.0
    )
    fig.suptitle("Corner plot — PDF marginali e correlazioni", fontsize=13, y=1.01)
    plt.savefig("corner_emcee.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Salvato: corner_emcee.png")

    # ── Diagramma di Hubble ──
    theta_med = np.median(flat_chain, axis=0)
    H0, Om, w0, wa = theta_med

    z_plot = np.linspace(0.01, 2.3, 300)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    # Banda di incertezza: campiona 300 punti dalla catena
    idx_s = np.random.choice(len(flat_chain), 300, replace=False)
    for idx in idx_s:
        H0s, Oms, w0s, was = flat_chain[idx]
        # Calcola M* ottimale per questo set di parametri
        mu_th  = mu_theory(z, H0s, Oms, w0s, was)
        # (qui usiamo solo la curva, non ricalcoliamo M per velocità)
        mu_plt = np.array([
            5*np.log10((c_light/H0s)*(1+zp)*quad(
                lambda zpp: 1/E(zpp, Oms, w0s, was), 0, zp, limit=50
            )[0]) + 25
            for zp in z_plot
        ])
        ax1.plot(z_plot, mu_plt, color="steelblue", alpha=0.03, lw=0.8)

    # Best fit
    mu_bf = np.array([
        5*np.log10((c_light/H0)*(1+zp)*quad(
            lambda zpp: 1/E(zpp, Om, w0, wa), 0, zp, limit=50
        )[0]) + 25
        for zp in z_plot
    ])
    ax1.plot(z_plot, mu_bf, "steelblue", lw=2,
             label=f"best fit  $H_0={H0:.1f}$, $\\Omega_m={Om:.3f}$")
    ax1.scatter(z, mu_obs, s=2, color="orange", alpha=0.3,
                zorder=5, label="Pantheon+")
    ax1.set_ylabel(r"$\mu$ (modulo di distanza)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_title("Diagramma di Hubble", fontsize=12)

    # Residui
    mu_pred = mu_theory(z, H0, Om, w0, wa)
    ones    = np.ones(len(z))
    from scipy.linalg import solve
    # Ricalcola M* per i residui
    cov_inv_diag = np.diag(np.ones(len(z)))  # approssimazione per il plot
    delta   = mu_obs - mu_pred
    ax2.scatter(z, delta - np.mean(delta), s=2, color="gray", alpha=0.4)
    ax2.axhline(0, color="steelblue", lw=1.5)
    ax2.set_xlabel("redshift $z$", fontsize=11)
    ax2.set_ylabel("residui", fontsize=11)
    ax2.set_ylim(-1.2, 1.2)

    plt.tight_layout()
    plt.savefig("hubble_emcee.png", dpi=150)
    plt.show()
    print("Salvato: hubble_emcee.png")

    return flat_chain


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    download_pantheon()
    z, mu_obs, cov_inv = load_data()

    sampler = run_emcee(
        z, mu_obs, cov_inv,
        n_walkers = 32,
        n_steps   = 3000,   # passi per walker dopo burn-in
        n_burnin  = 500     # passi di burn-in
    )
    # Totale valutazioni della likelihood: 32 * (500+3000) = 112000

    flat_chain = analyze_and_plot(sampler, z, mu_obs)