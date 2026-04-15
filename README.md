# MCMC_cosmology

Using the classic MCMC algorithm
## First: Only Supernovae Datasets
### INTRO:
Compilation of 1550 type-Ia supernovae across 18 surveys, covering the redshift range $0.001 < z < 2.26$. Goal was to map the hystory of the universe. expansion.. Uses Ia SNe as Standard Candles (objects with a known luminosity), meaning that if you see one SNe looking fainter than expected, then it must be further away than a simpler universe would expect.

MAIN PROBLEM: Pantheon alone can't give absolute measurement of the H0 without knowing the absolute M. They are degenerate!!

Solution: SH0ES (Supernovae on H0 for the Equation of State)  separate project with the goal of measuring the H0 in absolute sense, by anchoring the distance ladder. Using GAIA (Cepheid distances) they "measured"/calibrate the absolute M of the nearby SNe to get H0, breaking the degeneracy. Infere the M from cepheids charactristics. (they used only small redshifts!!!)

In SH0ES, the likelihood to introduce this calibration of M is not trivial. -> **to investigate!**

Results of SH0ES was to set $H0 = 73 \pm 1.04$. 

Summary: 
- Pantheon only ($z>0.01$) -> M, H0 degeneri
- Pantheon ($z>0.01$) + SH0ES ($z<0.01$) -> H0 "inferrable" but complicated since we need to treat differently the calibrated SN vs non-calibrated in the likelihood!
- Pantheon ($z>0.01$) + BAO (orthogonal dataset) -> H0 

### RUN
Potentially we want to estimate 5 parameters: $H0, M, \Omega _m, w0, wa$. In particular cases, estimate 5 parameters difficult for MCMC algorithm. 
- Esimation of H0, Ωm, M:
    - Only P, All free: BAD
    - Only P + marginalization of M: BAD still
    - Only p + Prior on H0 (from SH0ES), All free: Good results and good estimation of M (Ωm_Prior_Mfree)
- Estimation of H0, Ωm, M, w0, wa:
    - Prior + M marginal: (Ωmw0wa_Prior_Mmarginalized)
    - No prior: 

- Caso M marginal e prior:
  H0         = 73.18  +1.14 / -1.06
  Omega_m    = 0.33  +0.02 / -0.02

- Caso M free, prior H: (si vede chiaramente la ellisse per H0 e M)
  H0         = 73.1515  +0.8782 / -1.0058
  Omega_m    = 0.3289  +0.0192 / -0.0178
  M          = -0.0029  +0.0273 / -0.0329

- Caso M  marginal, prior e w:
  H0         = 72.9976  +1.1579 / -1.0501
  Omega_m    = 0.3227  +0.0706 / -0.0966
  w0         = -0.9439  +0.1369 / -0.1401
  wa         = -0.0915  +0.6585 / -1.1695

- Caso M free, prior H e w: Attenzione che però non so perchè ma in questo caso Ωm ha una distribuzione molto brutta e anche wa. Mentre w0 molto buona insieme a H0 e M
  H0         = 72.7461  +1.0267 / -0.8158
  Omega_m    = 0.3200  +0.0981 / -0.1354
  w0         = -0.9206  +0.1044 / -0.1485
  wa         = -0.1512  +0.9915 / -1.8467
  M          = -0.0131  +0.0299 / -0.0257


## Second: Supernovae + BAO
BAO is an orthogonal dataset wrt to SNe


Just for me:
+ SNe sole → H0 non misurabile, degenerazione totale
+ prior SH0ES → H0 ~ 73, M ~ 0, risultato pulito ma prior-dependent
+ prior Planck → H0 ~ 67, M ~ -0.1, tensione visibile
+ BAO, no prior → H0 ~ 68 dai dati, M ~ -0.16, tensione ancora presente
+ BAO, M marginalizzata → risultato più robusto, 4 parametri liberi