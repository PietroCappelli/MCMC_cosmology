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
import os

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


# ─────────────────────────────────────────────
# 3. LIKELIHOOD
# ─────────────────────────────────────────────

def log_likelihood(theta, z, mu_obs, cov_inv):
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
    H0, Omega_m, w0, wa, M = theta

    # Controllo sui parametri fisici (evita chiamate inutili)
    if Omega_m <= 0 or Omega_m >= 1:
        return -np.inf
    if H0 <= 0:
        return -np.inf

    mu_th = distance_modulus_model(z, H0, Omega_m, w0, wa)
    mu_th += M  # offset di calibrazione assoluta

    delta = mu_obs - mu_th
    return -0.5 * delta @ cov_inv @ delta


# ─────────────────────────────────────────────
# 4. PRIOR
# ─────────────────────────────────────────────

def log_prior(theta):
    """
    Prior piatte (uninformative) entro range fisicamente ragionevoli.
    Si possono sostituire con prior gaussiane se si hanno vincoli esterni.

    H0        in [50, 90]       km/s/Mpc
    Omega_m   in [0.1, 0.6]
    w0        in [-2, 0]        (w0 = -1 è LCDM)
    wa        in [-3, 3]
    M         in [-1, 1]        offset di calibrazione
    """
    H0, Omega_m, w0, wa, M = theta
    if (50 < H0 < 90 and
        0.1 < Omega_m < 0.6 and
        -2.0 < w0 < 0.0 and
        -3.0 < wa < 3.0 and
        -1.0 < M < 1.0):
        return 0.0  # log(1) = 0, prior piatta
    return -np.inf  # fuori dal range


def log_posterior(theta, z, mu_obs, cov_inv):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z, mu_obs, cov_inv)


# ─────────────────────────────────────────────
# 5. METROPOLIS-HASTINGS
# ─────────────────────────────────────────────

def run_mcmc(z, mu_obs, cov_inv,
             theta0,
             n_steps=10000,
             step_sizes=None,
             seed=42):
    """
    Algoritmo di Metropolis-Hastings.

    Parametri:
      theta0      : punto iniziale [H0, Omega_m, w0, wa, M]
      n_steps     : numero totale di passi
      step_sizes  : ampiezza della proposta gaussiana per ogni parametro
                    (da tuningare per avere acceptance rate ~23-40%)
      seed        : seme random per riproducibilità

    Restituisce:
      chain       : array (n_steps, n_params) con tutti i punti campionati
      log_post    : array (n_steps,) con i valori del log-posteriore
      accept_rate : frazione di mosse accettate
    """
    np.random.seed(seed)
    n_params = len(theta0)

    if step_sizes is None:
        # Step sizes di default (da tuningare!)
        # Regola pratica: ~1-2% del range del prior per ogni parametro
        step_sizes = np.array([0.5, 0.01, 0.05, 0.1, 0.01])

    chain    = np.zeros((n_steps, n_params))
    log_post = np.zeros(n_steps)

    theta_current  = np.array(theta0, dtype=float)
    lpost_current  = log_posterior(theta_current, z, mu_obs, cov_inv)
    n_accepted     = 0

    print(f"Avvio catena MCMC: {n_steps} passi, {n_params} parametri")
    print(f"Punto iniziale: {theta_current}")
    print(f"log-posterior iniziale: {lpost_current:.2f}")

    for i in range(n_steps):
        # Proposta: passo gaussiano simmetrico
        theta_proposed = theta_current + np.random.normal(0, step_sizes, n_params)

        # Calcola il log-posteriore nel nuovo punto
        lpost_proposed = log_posterior(theta_proposed, z, mu_obs, cov_inv)

        # Criterio di accettazione di Metropolis
        log_ratio = lpost_proposed - lpost_current
        if np.log(np.random.uniform()) < log_ratio:
            theta_current = theta_proposed
            lpost_current = lpost_proposed
            n_accepted += 1

        chain[i]    = theta_current
        log_post[i] = lpost_current

        # Progress ogni 1000 passi
        if (i + 1) % 1000 == 0:
            acc = n_accepted / (i + 1)
            print(f"  Passo {i+1}/{n_steps} — acceptance rate: {acc:.2%}")

    accept_rate = n_accepted / n_steps
    print(f"\nCatena completata. Acceptance rate finale: {accept_rate:.2%}")
    print("  (ideale: 23-40%. Se troppo basso -> riduci step_sizes, se troppo alto -> aumentali)")

    return chain, log_post, accept_rate


# ─────────────────────────────────────────────
# 6. ANALISI DELLA CATENA
# ─────────────────────────────────────────────

def analyze_chain(chain, log_post, burn_in_frac=0.3):
    """
    Rimuove il burn-in e calcola statistiche.

    burn_in_frac: frazione iniziale della catena da scartare (default 30%)
    """
    n_steps = len(chain)
    burn_in = int(n_steps * burn_in_frac)

    chain_burned = chain[burn_in:]
    print(f"\nBurn-in rimosso: {burn_in} passi. Catena utile: {len(chain_burned)} passi.")

    param_names = ["H0", "Omega_m", "w0", "wa", "M"]
    print("\nRisultati (mediana e intervallo 68%):")
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

def plot_trace(chain, log_post, burn_in_frac=0.3):
    """Trace plots: mostra l'evoluzione di ogni parametro nella catena."""
    param_names = ["H0", "Omega_m", "w0", "wa", "M"]
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
    plt.savefig("trace_plots.png", dpi=150)
    plt.show()
    print("Salvato: trace_plots.png")


def plot_corner(chain_burned):
    """
    Corner plot manuale (senza librerie esterne).
    Per un corner plot più bello: pip install corner
    e poi: import corner; corner.corner(chain_burned, labels=[...])
    """
    try:
        import corner
        param_names = ["H0", "Omega_m", "w0", "wa", "M"]
        fig = corner.corner(chain_burned, labels=param_names,
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, title_kwargs={"fontsize": 10})
        fig.suptitle("Corner plot — PDF marginali", fontsize=13)
        plt.savefig("corner_plot.png", dpi=150)
        plt.show()
        print("Salvato: corner_plot.png")
    except ImportError:
        print("Libreria 'corner' non installata. Installa con: pip install corner")
        print("Nel frattempo produco istogrammi 1D...")
        _plot_marginals(chain_burned)


def _plot_marginals(chain_burned):
    """Istogrammi 1D dei parametri (fallback se corner non è installato)."""
    param_names = ["H0", "Omega_m", "w0", "wa", "M"]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(chain_burned[:, i], bins=50, color="steelblue", edgecolor="white", lw=0.3)
        ax.set_xlabel(name, fontsize=10)
        ax.set_ylabel("conteggi", fontsize=9)
        ax.axvline(np.median(chain_burned[:, i]), color="orange", lw=1.5, label="mediana")
    axes[0].legend(fontsize=8)
    fig.suptitle("PDF marginali dei parametri", fontsize=12)
    plt.tight_layout()
    plt.savefig("marginals.png", dpi=150)
    plt.show()


def plot_hubble_diagram(z, mu_obs, chain_burned, n_samples=200):
    """
    Diagramma di Hubble: dati vs modello best-fit e banda di incertezza.
    """
    z_plot = np.linspace(0.01, 2.3, 300)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7),
                                    gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # Campiona n_samples punti dalla catena per la banda di incertezza
    idx_samples = np.random.choice(len(chain_burned), n_samples, replace=False)
    for idx in idx_samples:
        H0, Om, w0, wa, M = chain_burned[idx]
        mu_s = np.array([5 * np.log10(luminosity_distance(z, H0, Om, w0, wa)) + 25 + M
                         for z in z_plot])
        ax1.plot(z_plot, mu_s, color="steelblue", alpha=0.05, lw=0.8)

    # Best fit (mediana)
    theta_med = np.median(chain_burned, axis=0)
    H0, Om, w0, wa, M = theta_med
    mu_bf = np.array([5 * np.log10(luminosity_distance(z, H0, Om, w0, wa)) + 25 + M
                      for z in z_plot])
    ax1.plot(z_plot, mu_bf, color="steelblue", lw=2, label=f"best fit")

    # Dati
    ax1.scatter(z, mu_obs, s=3, color="orange", alpha=0.4, zorder=5, label="Pantheon+")
    ax1.set_ylabel(r"$\mu$ (modulo di distanza)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_title("Diagramma di Hubble", fontsize=12)

    # Residui
    mu_pred = distance_modulus_model(z, H0, Om, w0, wa) + M
    residuals = mu_obs - mu_pred
    ax2.scatter(z, residuals, s=3, color="gray", alpha=0.4)
    ax2.axhline(0, color="steelblue", lw=1.5)
    ax2.set_xlabel("redshift z", fontsize=11)
    ax2.set_ylabel("residui", fontsize=11)
    ax2.set_ylim(-1.5, 1.5)

    plt.tight_layout()
    plt.savefig("hubble_diagram.png", dpi=150)
    plt.show()
    print("Salvato: hubble_diagram.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Scarica e carica i dati
    download_pantheon()
    z, mu_obs, cov_inv = load_data()

    # 2. Punto di partenza della catena
    # [H0,   Omega_m, w0,   wa,  M  ]
    theta0 = [70.0,  0.30,   -1.0, 0.0, 0.0]

    # 3. Esegui l'MCMC
    # NOTA: per un risultato affidabile usa n_steps >= 50000
    # Per un test rapido inizia con n_steps=5000
    chain, log_post, acc_rate = run_mcmc(
        z, mu_obs, cov_inv,
        theta0=theta0,
        n_steps=20000,
        step_sizes=[0.4, 0.008, 0.04, 0.08, 0.008]
    )

    # 4. Analisi della catena (rimuove burn-in)
    chain_burned, results = analyze_chain(chain, log_post, burn_in_frac=0.3)

    # 5. Plot
    plot_trace(chain, log_post)
    plot_corner(chain_burned)
    plot_hubble_diagram(z, mu_obs, chain_burned)
    