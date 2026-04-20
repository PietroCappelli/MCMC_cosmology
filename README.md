# MCMC_cosmology
PAPERS:
- Pantheon: https://arxiv.org/pdf/2202.04077
- SH0ES: https://arxiv.org/pdf/2404.03002
- BAO: https://pantheonplussh0es.github.io/
- Marginalization of M: https://arxiv.org/abs/astro-ph/0104009
extra:
- DESI collaboration website cute for images: https://data.desi.lbl.gov/public/

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

- case M marginal e prior:
  H0         = 73.18  +1.14 / -1.06
  Omega_m    = 0.33  +0.02 / -0.02

- case M free, prior H: (si vede chiaramente la ellisse per H0 e M)
  H0         = 73.1515  +0.8782 / -1.0058
  Omega_m    = 0.3289  +0.0192 / -0.0178
  M          = -0.0029  +0.0273 / -0.0329

- case M  marginal, prior e w:
  H0         = 72.9976  +1.1579 / -1.0501
  Omega_m    = 0.3227  +0.0706 / -0.0966
  w0         = -0.9439  +0.1369 / -0.1401
  wa         = -0.0915  +0.6585 / -1.1695

- case M free, prior H e w: Attenzione che però non so perchè ma in questo case Ωm ha una distribuzione molto brutta e anche wa. Mentre w0 molto buona insieme a H0 e M
  H0         = 72.7461  +1.0267 / -0.8158
  Omega_m    = 0.3200  +0.0981 / -0.1354
  w0         = -0.9206  +0.1044 / -0.1485
  wa         = -0.1512  +0.9915 / -1.8467
  M          = -0.0131  +0.0299 / -0.0257


## Second: Supernovae + BAO
BAO is an orthogonal dataset wrt to SNe


## Just for me (Pietro):
+ SNe sole → H0 non misurabile, degenerazione totale
+ prior SH0ES → H0 ~ 73, M ~ 0, risultato pulito ma prior-dependent
+ prior Planck → H0 ~ 67, M ~ -0.1, tensione visibile
+ BAO, no prior → H0 ~ 68 dai dati, M ~ -0.16, tensione ancora presente
+ BAO, M marginal → risultato più robusto, 4 Parameters liberi


- Block 1 — Solo SNe, understand degenerazioni
    - case 1a — SNe, ΛCDM, M libera, no prior
    Parameters: H0, Ω_m, M
    Risultato atteso: H0 vaga, M degenere con H0, Ω_m ok
    Messaggio: degenerazione H0-M, le SNe non misurano H0
    ```
    H0         = 71.9763  +6.3445 / -2.5632
    Omega_m    = 0.3302  +0.0190 / -0.0192
    M          = -0.0398  +0.1857 / -0.0788
    ```

    - case 1b — SNe, ΛCDM, M marginal, no prior
    Parameters: H0, Ω_m
    Risultato atteso: H0 ancora vaga ma Ω_m pulito
    Messaggio: marginalizzare M non basta, serve ancora un'ancora su H0
    ```
    H0         = 77.5700  +10.2513 / -9.3145
    Omega_m    = 0.3313  +0.0183 / -0.0191
    ```

    - case 1c — SNe, ΛCDM, M free, prior Planck
    Parameters: H0, Ω_m
    Risultato atteso: H0 ~ 67.4, Ω_m ~ 0.33, distribuzioni pulite
    Messaggio: con prior esterno tutto converge — ma stai assumendo Planck
    ```
    H0         = 67.4471  +0.4544 / -0.4800
    Omega_m    = 0.3314  +0.0166 / -0.0179
    M          = -0.1790  +0.0158 / -0.0174
    ```

    - case 1d — SNe, ΛCDM, M marginal, prior SH0ES
    Parameters: H0, Ω_m
    Risultato atteso: H0 ~ 73, Ω_m leggermente diverso da 1c
    Messaggio: prior diversa → risultato diverso → questa è la tensione di Hubble
    ```
    H0         = 73.0620  +1.0153 / -1.0774
    Omega_m    = 0.3322  +0.0182 / -0.0186
    ```

    - case 1e — SNe, ΛCDM, M free, prior SH0ES
    Parameters: H0, Ω_m, M
    ```
  H0         = 73.2458  +1.0085 / -0.9963
  Omega_m    = 0.3307  +0.0190 / -0.0169
  M          = -0.0001  +0.0313 / -0.0304
    ```


Con M libera la posterior è allungata lungo la direzione H0-M — la catena deve fare passi grandi per esplorare quella banana. Con M marginal quella dimensione è eliminata e la posterior è molto più compatta — gli stessi step sizes diventano relativamente piccoli rispetto alla nuova geometria. Per questo devi aumentarli.

- Block 2 — Aggiungere BAO, rompere le degenerazioni
    - case 2a — SNe + BAO, ΛCDM, M libera, no prior
    Parameters: H0, Ω_m, M
    Risultato atteso: H0 ~ 68 dai dati, M ~ -0.16, Ω_m ~ 0.30
    Messaggio: i BAO ancorano H0 geometricamente senza prior — M ≠ 0 è la tensione di Hubble
    ```
    H0         = 68.5468  +0.6795 / -0.6360
    Omega_m    = 0.3093  +0.0100 / -0.0113
    M          = -0.1512  +0.0185 / -0.0169
    ```
    - case 2b — SNe + BAO, ΛCDM, M marginal, no prior
    Parameters: H0, Ω_m
    Risultato atteso: H0 ~ 68, Ω_m ~ 0.30, distribuzioni pulite
    Messaggio: case più robusto per ΛCDM — nessuna assunzione esterna
    ```
    H0         = 68.3901  +0.7592 / -0.7499
    Omega_m    = 0.3114  +0.0125 / -0.0127
    ```
  
- Block 3 — Energia oscura dinamica
    - case 3a — SNe + BAO, w0CDM, M marginal, no prior
    Parameters: H0, Ω_m, w0
    Risultato atteso: w0 ~ -0.9, compatibile con -1
    Messaggio: primo test su energia oscura, le SNe+BAO sono compatibili con ΛCDM
    ```
    H0         = 68.0834  +0.7820 / -0.7608
    Omega_m    = 0.2972  +0.0143 / -0.0139
    w0         = -0.9237  +0.0468 / -0.0476
    ```
    - case 3b — SNe + BAO, w0waCDM, M marginal, no prior
    Parameters: H0, Ω_m, w0, wa
    Risultato atteso: w0 ~ -0.89, wa ~ -0.33 con grandi incertezze
    Messaggio: wa mal vincolato senza CMB, ma compatibile con 0 — no evidenza forte di DE dinamica
    ```
    H0         = 68.0651  +0.7925 / -0.7226
    Omega_m    = 0.3130  +0.0167 / -0.0202
    w0         = -0.8714  +0.0696 / -0.0690
    wa         = -0.5480  +0.5856 / -0.5330
    ```

- Bonus:
    - SNe only, w0waCDM, M marginal, prior Planck 
    → mostra quanto sono larghe le posterior su w0 e wa senza BAO
    ```
    H0         = 72.9334  +1.0545 / -0.9754
    Omega_m    = 0.3195  +0.0816 / -0.0984
    w0         = -0.9344  +0.1249 / -0.1428
    wa         = -0.1347  +0.7599 / -1.4436
    ```

    - SNe + BAO, w0waCDM, M marginal, no prior 
    → stesso modello con BAO → le posterior si stringono visibilmente

    
(Ha senso includerlo solo come case dimostrativo per mostrare cosa succede senza BAO. Il messaggio sarebbe chiaro — w0 e wa sono completamente mal vincolati con sole SNe, le posterior sono larghissime e la banana w0-wa è enorme.
Però attenzione — senza BAO e senza prior su H0, con 4 Parameters liberi la catena faticherà molto a convergere. Aggiungi almeno il prior su H0, altrimenti rischi di non ottenere nulla di interpretabile.)

I plot da fare
Plot 1 — Corner plot per ogni case
Mostrano le posterior marginali e le correlazioni tra Parameters. I più importanti da confrontare sono 1c vs 1d (tensione di Hubble) e 2b vs 3b (effetto di aggiungere w0, wa).
Plot 2 — Confronto posterior H0
Un singolo plot con tutte le distribuzioni marginali di H0 sovrapposte — casi 1c, 1d, 2a, 2b. Si vede visivamente come il prior o i BAO spostano e stringono H0.
Plot 3 — Diagramma di Hubble
Best fit + banda di incertezza sovrapposta ai dati Pantheon. Uno per il case ΛCDM e uno per w0waCDM — si vede come le curve differiscono a alto redshift.
Plot 4 — Confronto contorni w0-wa
Se hai il tempo, un plot che mostra i contorni 68% e 95% nel piano w0-wa con una stella su (-1, 0) per ΛCDM. È la figura classica dei paper di energia oscura.