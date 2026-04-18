"""
MCMC Likelihood Analysis — Pantheon+ Supernovae
================================================
Parametri stimati: H0, Omega_m, w0, wa, M (offset assoluto), sigma_int (scatter intrinseco)
Dataset: Pantheon+SH0ES (~1590 SNe dopo il taglio z > 0.01)

Struttura del codice:
  1. Download e caricamento del dataset
  2. Modello cosmologico: d_L(z, theta) -> mu_model
  3. Likelihood ln L(theta)
  4. Prior ln P(theta)
  5. Algoritmo Metropolis-Hastings
  6. Analisi della catena: burn-in, autocorrelazione
  7. Plot: trace plots, corner plot, best-fit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pandas as pd
import requests
import sys
import os
import emcee

# ── Select the model ──────────────────────
MODEL = "LCDM_Mmarg_NoPrior_w0wa"
# ──────────────────────────────────────────────

MODELS = {
    "LCDM_Mfree_NoPrior": {
        "params":     ["H0", "Omega_m", "M"],
        "fixed":      {"w0": -1.0, "wa": 0.0},
        "theta0":     [70.0, 0.30, 0.0],
        "step_sizes": [0.4, 0.008, 0.008],
        "prior_bounds": {
            "H0": (50, 90),
            "Omega_m": (0.1, 0.6),
            "M": (-1.0, 1.0)
        },
        "prior_gauss":  {},
        "marginalize_M": False,
        "n_steps": 30000
    },
    "LCDM_Mmarg_NoPrior": {
        "params":     ["H0", "Omega_m"],
        "fixed":      {"w0": -1.0, "wa": 0.0},
        "theta0":     [70.0, 0.30],
        "step_sizes": [0.4, 0.008],
        "prior_bounds": {
            "H0": (50, 90),
            "Omega_m": (0.1, 0.6),
        },
        "prior_gauss":  {},
        "marginalize_M": True,
        "n_steps": 30000
    },
    "LCDM_Mmarg_NoPrior_w0": {
        "params":     ["H0", "Omega_m", "w0"],
        "fixed":      {"wa": 0.0},
        "theta0":     [70.0, 0.30, -1.0],
        "step_sizes": [0.4, 0.008, 0.05],
        "prior_bounds": {
            "H0": (50, 90),
            "Omega_m": (0.1, 0.6),
            "w0": (-2, 0)},
        "prior_gauss":  {},
        "marginalize_M": True,
        "n_steps": 30000
    },
    "LCDM_Mmarg_NoPrior_w0wa": {
        "params":     ["H0", "Omega_m", "w0", "wa"],
        "fixed":      {},
        "theta0":     [70.0, 0.30, -1.0, 0.0],
        "step_sizes": [0.4, 0.008, 0.05, 0.1],
        "prior_bounds": {
            "H0": (50, 90),
            "Omega_m": (0.1, 0.6),
            "w0": (-2, 0),
            "wa": (-3, 3)},
        "prior_gauss":  {},
        "marginalize_M": True,
    }
}

# ─────────────────────────────────────────────
# 1. DOWNLOAD E CARICAMENTO DEL DATASET
# ─────────────────────────────────────────────

def download_pantheon(data_path="Pantheon+SH0ES.dat", cov_path="Pantheon+SH0ES_STAT+SYS.cov"):
    base = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
    if not os.path.exists(data_path):
        print("Scaricando il file dati...")
        r = requests.get(base + "Pantheon%2BSH0ES.dat")
        with open(data_path, "w") as f:
            f.write(r.text)
    if not os.path.exists(cov_path):
        print("Scaricando la matrice di covarianza...")
        r = requests.get(base + "Pantheon%2BSH0ES_STAT%2BSYS.cov")
        with open(cov_path, "w") as f:
            f.write(r.text)
    print("Dataset pronto.")

def load_data(data_path="Pantheon+SH0ES.dat", cov_path="Pantheon+SH0ES_STAT+SYS.cov"):
    # Carica il file principale
    df = pd.read_csv(data_path, sep=r'\s+')

    # Filtro cosmologico: escludi SNe locali usate da SH0ES come calibratori
    # zHD > 0.01 rimuove le SNe con Cefeidi, lascia ~1590 SNe
    mask = df["zHD"] > 0.01
    df = df[mask].reset_index(drop=True)

    z       = df["zHD"].values          # redshift
    mu_obs  = df["MU_SH0ES"].values     # modulo di distanza osservato
    # mu_err  = df["MU_SH0ES_ERR_DIAG"].values  # solo diagonale (non usata se usi la cov completa)

    n = len(z)
    print(f"Supernovae caricate dopo il taglio z>0.01: {n}")

    # Carica la matrice di covarianza completa N x N
    with open(cov_path) as f:
        n_cov = int(f.readline())
        cov_flat = np.array(f.read().split(), dtype=float)
    cov_full = cov_flat.reshape(n_cov, n_cov)

    # Applica lo stesso filtro alla matrice di covarianza
    # (il file .cov ha lo stesso ordine delle righe del .dat PRIMA del filtro)
    df_full = pd.read_csv(data_path, sep=r'\s+')
    mask_full = df_full["zHD"] > 0.01
    idx = np.where(mask_full.values)[0]
    cov = cov_full[np.ix_(idx, idx)]

    # Inverti la matrice una volta sola (costoso, non farlo ad ogni step!)
    cov_inv = np.linalg.inv(cov)

    return z, mu_obs, cov_inv

BAO_DATA = [
    # z      DM/rd   sDM    DH/rd   sDH    rho
    (0.510,  13.62,  0.25,  20.98,  0.61,  -0.44),  # LRG1
    (0.706,  16.85,  0.32,  20.08,  0.60,  -0.35),  # LRG2
    (0.930,  21.71,  0.28,  17.88,  0.35,  -0.38),  # LRG3+ELG1
    (1.317,  27.79,  0.69,  13.82,  0.42,  -0.47),  # ELG2
    (2.330,  39.71,  0.94,   8.52,  0.17,  -0.45),  # Lya
]
# Punti con solo DV/rd (isotropici) — BGS e QSO
BAO_DATA_ISO = [
    # z      DV/rd   sDV
    (0.295,  7.93,   0.15),   # BGS
    (1.491,  26.07,  0.67),   # QSO
]
# ─────────────────────────────────────────────
# 2. MODELLO COSMOLOGICO
# ─────────────────────────────────────────────

c_light = 2.998e5  # km/s

def E(z, Omega_m, w0, wa):
    """
    Funzione di Hubble adimensionale E(z) = H(z)/H0.
    Modello w0wa: energia oscura con equazione di stato w(z) = w0 + wa * z/(1+z).
    Con w0=-1, wa=0 si riduce al LCDM standard.
    """
    Omega_de = 1.0 - Omega_m  # universo piatto: Omega_k = 0
    # Fattore di densità per energia oscura con w(z) variabile
    # f_de(z) = exp(3 * integral_0^z [1 + w(z')]/(1+z') dz')
    # Per w0wa si integra analiticamente:
    f_de = (1 + z)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_de * f_de)

def luminosity_distance(z, H0, Omega_m, w0, wa):
    """
    Distanza di luminosità in Mpc.
    d_L(z) = c(1+z)/H0 * integral_0^z dz'/E(z')
    """
    def integrand(zp):
        return 1.0 / E(zp, Omega_m, w0, wa)

    integral, _ = quad(integrand, 0, z, limit=100)
    return (c_light / H0) * (1 + z) * integral

def distance_modulus_model(z_array, H0, Omega_m, w0, wa):
    """
    Calcola mu_model per un array di redshift.
    mu = 5 * log10(d_L / 10pc) = 5 * log10(d_L in Mpc) + 25
    """
    mu = np.zeros(len(z_array))
    for i, z in enumerate(z_array):
        dL = luminosity_distance(z, H0, Omega_m, w0, wa)  # in Mpc
        mu[i] = 5 * np.log10(dL) + 25  # converti Mpc -> pc aggiunge 25
    return mu

def comoving_distance(z, H0, Omega_m, w0, wa):
    """Distanza comovente in Mpc."""
    def integrand(zp):
        return 1.0 / E(zp, Omega_m, w0, wa)
    integral, _ = quad(integrand, 0, z, limit=100)
    return (c_light / H0) * integral

def DM_over_rd(z, H0, Omega_m, w0, wa, rd=147.09):
    return comoving_distance(z, H0, Omega_m, w0, wa) / rd

def DH_over_rd(z, H0, Omega_m, w0, wa, rd=147.09):
    return (c_light / (H0 * E(z, Omega_m, w0, wa))) / rd

# ─────────────────────────────────────────────
# 3. LIKELIHOOD
# ─────────────────────────────────────────────

def log_likelihood_sne(theta, z, mu_obs, cov_inv, model_cfg):
    """
    ln L(theta) = -1/2 * Delta_mu^T C^{-1} Delta_mu

    theta = [H0, Omega_m, w0, wa, M, sigma_int]

    M è un offset sulla calibrazione assoluta:
      mu_model_corrected = mu_model + M
    sigma_int è lo scatter intrinseco delle SNe Ia:
      aggiunge sigma_int^2 alla diagonale della covarianza
      (in questa implementazione semplificata lo ignoriamo,
       ma è il settimo parametro standard nelle analisi avanzate)
    """
    # unpack dinamico
    params = dict(zip(model_cfg["params"], theta))
    params.update(model_cfg["fixed"]) # aggiungi i parametri fissati

    H0, Omega_m = params["H0"], params["Omega_m"]
    w0, wa, M   = params["w0"], params["wa"], params["M"]

    # Controllo sui parametri fisici (evita chiamate inutili)
    if Omega_m <= 0 or Omega_m >= 1:
        return -np.inf
    if H0 <= 0:
        return -np.inf

    mu_th = distance_modulus_model(z, H0, Omega_m, w0, wa)
    mu_th += M  # offset di calibrazione assoluta

    delta = mu_obs - mu_th
    return -0.5 * delta @ cov_inv @ delta

def log_likelihood_sne_marginal(theta, z, mu_obs, cov_inv, model_cfg):
    params = dict(zip(model_cfg["params"], theta))
    params.update(model_cfg["fixed"])

    H0      = params["H0"]
    Omega_m = params["Omega_m"]
    w0      = params["w0"]
    wa      = params["wa"]

    if Omega_m <= 0 or Omega_m >= 1:
        return -np.inf
    if H0 <= 0:
        return -np.inf

    mu_th = distance_modulus_model(z, H0, Omega_m, w0, wa)
    delta = mu_obs - mu_th  # niente M

    # Marginalizzazione analitica su M
    A = delta @ cov_inv @ delta
    B = np.sum(cov_inv @ delta)
    C = np.sum(cov_inv)

    return -0.5 * (A - B**2 / C)

def log_likelihood_bao(theta, model_cfg, bao_data=BAO_DATA):
    params = dict(zip(model_cfg["params"], theta))
    params.update(model_cfg["fixed"])

    H0      = params["H0"]
    Omega_m = params["Omega_m"]
    w0      = params["w0"]
    wa      = params["wa"]

    if Omega_m <= 0 or Omega_m >= 1 or H0 <= 0:
        return -np.inf

    log_l = 0.0
    for (z, DM_obs, sDM, DH_obs, sDH, rho) in bao_data:

        DM_th = DM_over_rd(z, H0, Omega_m, w0, wa)
        DH_th = DH_over_rd(z, H0, Omega_m, w0, wa)

        # Residui
        delta = np.array([DM_obs - DM_th, DH_obs - DH_th])

        # Matrice di covarianza 2x2 per questo punto
        cov_bao = np.array([
            [sDM**2,       rho * sDM * sDH],
            [rho * sDM * sDH, sDH**2      ]
        ])
        cov_bao_inv = np.linalg.inv(cov_bao)

        log_l += -0.5 * delta @ cov_bao_inv @ delta

    return log_l

def log_likelihood_combined(theta, z_sne, mu_obs, cov_inv, model_cfg):
    if model_cfg.get("marginalize_M", False):  # default False if no key
        sne_ll = log_likelihood_sne_marginal(theta, z_sne, mu_obs, cov_inv, model_cfg)
    else:
        sne_ll = log_likelihood_sne(theta, z_sne, mu_obs, cov_inv, model_cfg)
    
    return sne_ll + log_likelihood_bao(theta, model_cfg)

# ─────────────────────────────────────────────
# 4. PRIOR
# ─────────────────────────────────────────────

def log_prior(theta, model_cfg):
    params = dict(zip(model_cfg["params"], theta))
    bounds = model_cfg["prior_bounds"]

    # Prior uniform
    for name, value in params.items():
        lo, hi = bounds[name]
        if not (lo < value < hi):
            return -np.inf

    # Prior gaussian
    lp = 0.0
    for name, (mean, std) in model_cfg.get("prior_gauss", {}).items():
        if name in params:
            lp += -0.5 * ((params[name] - mean) / std)**2

    return lp

# ─────────────────────────────────────────────
# 4. FAKE-POSTERIOR
# ─────────────────────────────────────────────
def log_posterior(theta, z_sne, mu_obs, cov_inv, model_cfg):
    lp = log_prior(theta, model_cfg)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_combined(theta, z_sne, mu_obs, cov_inv, model_cfg)

# ─────────────────────────────────────────────
# 5. METROPOLIS-HASTINGS
# ─────────────────────────────────────────────

def run_mcmc_emcee(z, mu_obs, cov_inv, model_cfg, n_steps=5000, n_walkers=32):
    
    n_params  = len(model_cfg["params"])
    theta0    = np.array(model_cfg["theta0"])
    
    # Inizializza i walker con piccola perturbazione intorno a theta0
    pos = theta0 + 1e-3 * np.random.randn(n_walkers, n_params)
    
    # Funzione log_posterior già esistente — non cambia nulla!
    def log_prob(theta):
        return log_posterior(theta, z, mu_obs, cov_inv, model_cfg)
    
    sampler = emcee.EnsembleSampler(n_walkers, n_params, log_prob)
    
    # Burn-in
    print("Burn-in...")
    pos, _, _ = sampler.run_mcmc(pos, 500, progress=True)
    sampler.reset()
    
    # Run principale
    print("Run Main...")
    sampler.run_mcmc(pos, n_steps, progress=True)
    
    # Estrai la chain — shape (n_walkers * n_steps, n_params)
    chain = sampler.get_chain(flat=True)
    log_post = sampler.get_log_prob(flat=True)
    
    print(f"Average acceptance rate: {np.mean(sampler.acceptance_fraction):.2%}")
    
    return chain, log_post

# ─────────────────────────────────────────────
# 6. ANALISI DELLA CATENA
# ─────────────────────────────────────────────

def analyze_chain(chain, log_post, model_cfg, burn_in_frac=0.3):
    """
    Rimuove il burn-in e calcola statistiche.

    burn_in_frac: frazione iniziale della catena da scartare (default 30%)
    """
    n_steps = len(chain)
    burn_in = int(n_steps * burn_in_frac)

    chain_burned = chain[burn_in:]
    print(f"\nBurn-in removed: {burn_in} steps. Used chain: {len(chain_burned)} steps.")

    # param_names = ["H0", "Omega_m", "w0", "wa", "M"]
    param_names = model_cfg["params"]
    
    
    print("\nResults (median and interval 68%):")
    print("-" * 45)
    results = {}
    for i, name in enumerate(param_names):
        samples = chain_burned[:, i]
        med     = np.median(samples)
        lo      = np.percentile(samples, 16)
        hi      = np.percentile(samples, 84)
        print(f"  {name:10s} = {med:.4f}  +{hi-med:.4f} / -{med-lo:.4f}")
        results[name] = (med, lo, hi)

    return chain_burned, results


# ─────────────────────────────────────────────
# 7. PLOT
# ─────────────────────────────────────────────

def plot_trace(chain, log_post, model_cfg, folder_name, burn_in_frac=0.3):
    """Trace plots: mostra l'evoluzione di ogni parametro nella catena."""
    # param_names = ["H0", "Omega_m", "w0", "wa", "M"]
    param_names = model_cfg["params"]
    
    n_steps = len(chain)
    burn_in = int(n_steps * burn_in_frac)

    fig, axes = plt.subplots(len(param_names) + 1, 1, figsize=(10, 12), sharex=True)

    for i, (ax, name) in enumerate(zip(axes[:-1], param_names)):
        ax.plot(chain[:, i], color="steelblue", lw=0.4, alpha=0.8)
        ax.axvline(burn_in, color="red", lw=1.2, linestyle="--", label="burn-in")
        ax.set_ylabel(name, fontsize=10)
        ax.tick_params(labelsize=8)

    axes[-1].plot(log_post, color="gray", lw=0.4)
    axes[-1].axvline(burn_in, color="red", lw=1.2, linestyle="--")
    axes[-1].set_ylabel("ln posterior", fontsize=10)
    axes[-1].set_xlabel("passo MCMC", fontsize=10)

    axes[0].legend(fontsize=9)
    fig.suptitle("Trace plots", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{folder_name}/trace_plots.png", dpi=150)
    plt.show()
    print("Salvato: trace_plots.png")


def plot_corner(chain_burned, model_cfg, folder_name):
    try:
        import corner

        param_names = model_cfg["params"]

        # Palette colori
        color_main  = "#2E86AB"   # blu acciaio
        color_fill  = "#A8DADC"   # azzurro chiaro
        color_quant = "#E63946"   # rosso vivo per i quantili

        fig = corner.corner(
            chain_burned,
            labels=param_names,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 11, "color": color_main},
            label_kwargs={"fontsize": 12},
            color=color_main,
            levels=(0.68, 0.95),
            fill_contours=True,
            contourf_kwargs={"colors": [color_fill, color_main], "alpha": 0.5},
            contour_kwargs={"colors": color_main, "linewidths": 1.2},
            hist_kwargs={"color": color_main, "linewidth": 1.8,
                         "histtype": "stepfilled", "alpha": 0.35,
                         "edgecolor": color_main},
            quantile_kwargs={"color": color_quant, "linewidth": 1.2, "linestyle": "--"},
        )

        fig.patch.set_facecolor("white")
        for ax in fig.get_axes():
            ax.set_facecolor("white")
            ax.tick_params(colors="#333333", labelsize=8)
            ax.xaxis.label.set_color("#333333")
            ax.yaxis.label.set_color("#333333")
            for spine in ax.spines.values():
                spine.set_edgecolor("#CCCCCC")  # bordi pannelli grigi sottili

        fig.suptitle("Corner Plot — Marginali PDF",
                     fontsize=14, color="#333333",
                     fontweight="bold", y=1.01)

        plt.savefig(f"{folder_name}/corner_plot.png",
                    dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.show()
        print("Salvato: corner_plot.png")

    except ImportError:
        print("Libreria 'corner' non installata. Installa con: pip install corner")
        print("Nel frattempo produco istogrammi 1D...")
        _plot_marginals(chain_burned)

    _plot_marginals(chain_burned, model_cfg, folder_name)
    

def _plot_marginals(chain_burned, model_cfg, folder_name):
    """Istogrammi 1D dei parametri (fallback se corner non è installato)."""
    # param_names = ["H0", "Omega_m", "w0", "wa", "M"]
    param_names = model_cfg["params"]
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(chain_burned[:, i], bins=50, color="steelblue", edgecolor="white", lw=0.3)
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel("conteggi", fontsize=9)
        ax.axvline(np.median(chain_burned[:, i]), color="orange", lw=1.5, label="mediana")
    axes[0].legend(fontsize=8)
    fig.suptitle("PDF marginali dei parametri", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{folder_name}/marginals.png", dpi=150)
    plt.show()


def plot_hubble_diagram(z, mu_obs, chain_burned, model_cfg, folder_name, n_samples=200):
    
    z_plot     = np.linspace(0.01, 2.3, 300)
    param_names = model_cfg["params"]
    fixed       = model_cfg["fixed"]

    def get_params(theta):
        """Ricostruisce il dizionario completo params + fixed."""
        params = dict(zip(param_names, theta))
        params.update(fixed)
        return (params["H0"], params["Omega_m"],
                params["w0"], params["wa"], params.get("M", 0.0))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7),
                                    gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # Banda di incertezza
    idx_samples = np.random.choice(len(chain_burned), n_samples, replace=False)
    for idx in idx_samples:
        H0, Om, w0, wa, M = get_params(chain_burned[idx])
        mu_s = np.array([5 * np.log10(luminosity_distance(zz, H0, Om, w0, wa)) + 25 + M
                         for zz in z_plot])
        ax1.plot(z_plot, mu_s, color="steelblue", alpha=0.05, lw=0.8)

    # Best fit (mediana)
    theta_med          = np.median(chain_burned, axis=0)
    H0, Om, w0, wa, M  = get_params(theta_med)
    mu_bf = np.array([5 * np.log10(luminosity_distance(zz, H0, Om, w0, wa)) + 25 + M
                      for zz in z_plot])
    ax1.plot(z_plot, mu_bf, color="steelblue", lw=2, label="best fit")

    ax1.scatter(z, mu_obs, s=3, color="orange", alpha=0.4, zorder=5, label="Pantheon+")
    ax1.set_ylabel(r"$\mu$ (modulo di distanza)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_title("Diagramma di Hubble", fontsize=12)

    # Residui
    mu_pred   = distance_modulus_model(z, H0, Om, w0, wa) + M
    residuals = mu_obs - mu_pred
    ax2.scatter(z, residuals, s=3, color="gray", alpha=0.4)
    ax2.axhline(0, color="steelblue", lw=1.5)
    ax2.set_xlabel("redshift z", fontsize=11)
    ax2.set_ylabel("residui", fontsize=11)
    ax2.set_ylim(-1.5, 1.5)

    plt.tight_layout()
    plt.savefig(f"{folder_name}/hubble_diagram.png", dpi=150)
    plt.show()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python script.py <folder_name>")
        sys.exit(1)

    folder_name = sys.argv[1]
    os.makedirs(folder_name, exist_ok=True)
    
    # 1. Scarica e carica i dati
    # download_pantheon()
    z, mu_obs, cov_inv = load_data()

    # 2. Punto di partenza della catena
    cfg = MODELS[MODEL]
    theta0 = cfg["theta0"]

    # 3. Esegui l'MCMC
    chain, log_post = run_mcmc_emcee(
        z, mu_obs, cov_inv, 
        model_cfg=cfg,
        n_steps=5000, 
        n_walkers=32
        )
    
    # 4. Analisi della catena (rimuove burn-in)
    chain_burned, results = analyze_chain(chain, log_post, burn_in_frac=0.0, model_cfg=cfg)

    # 5. Plot
    plot_trace(chain, log_post, folder_name=folder_name, model_cfg=cfg)
    plot_corner(chain_burned, folder_name=folder_name, model_cfg=cfg)
    plot_hubble_diagram(z, mu_obs, chain_burned, folder_name=folder_name, model_cfg=cfg)
    