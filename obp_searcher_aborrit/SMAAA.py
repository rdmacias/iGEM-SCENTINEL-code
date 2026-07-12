import numpy as np


# ─────────────────────────────────────────────
# GENERADORS DE PESOS
# ─────────────────────────────────────────────

def _generar_pesos_salts(rng, n_iter, pesos_base, c):
    """
    Mètode de salts: descomposició en terra (valor comú dels dos últims criteris)
    + salts (gaps entre criteris consecutius). Cada peça balla uniformement dins
    de ±c% del seu valor base; llavors es reconstrueix i normalitza.

    c=0  → pesos fixos; c=100 → cada peça pot doblar o anul·lar-se.

    Restricció: els dos últims criteris (estabilitat = promiscuïtat) es forcen
    a la seva mitjana i sempre resten iguals (salt entre ells = 0).
    L'ordre w[0] ≥ w[1] ≥ … ≥ w[-2] = w[-1] és necessari per construcció — com
    que pesos_base pot venir de l'usuari (no necessàriament ordenat), els
    criteris davanters (tots excepte els 2 últims) es reordenen de gran a
    petit i es retallen al terra abans de generar, i les columnes de sortida
    es tornen a l'ordre original perquè coincideixin amb subscores/te_dada.
    Sense això, un pes davanter per sota del terra dona rangs uniform(lo>hi)
    invertits — comportament no definit a NumPy — i pot produir pesos negatius.
    """
    base = np.array(pesos_base, dtype=float)
    base = base / base.sum()
    base[-2] = base[-1] = (base[-2] + base[-1]) / 2.0

    k = len(base)
    f = c / 100.0

    if k > 2:
        ordre     = np.argsort(-base[:-2], kind='stable')
        inv_ordre = np.argsort(ordre)
        base[:-2] = np.maximum(base[:-2][ordre], base[-1])

    # terra = valor comú dels dos últims criteris
    terra = base[-1]
    # salts[i] = base[i] - base[i+1] > 0, excloent l'últim par (sempre 0)
    salts = (-np.diff(base))[:-1]  # (k-2,)

    # Terra balla ±c% (clipat a positiu mínim)
    lo_t = max(terra * (1.0 - f), 1e-9)
    hi_t = terra * (1.0 + f)
    terra_s = rng.uniform(lo_t, hi_t, size=n_iter)  # (n_iter,)

    w_cru = np.empty((n_iter, k))
    w_cru[:, -2] = terra_s
    w_cru[:, -1] = terra_s

    if k > 2:
        # Cada salt balla ±c% (clipat a 0 per sota — no pot fer-se negatiu)
        lo_salts = np.maximum(salts * (1.0 - f), 0.0)  # (k-2,)
        hi_salts = salts * (1.0 + f)                    # (k-2,)
        u = rng.random((n_iter, k - 2))
        salt_s = lo_salts + u * (hi_salts - lo_salts)   # (n_iter, k-2)

        # w[:,i] = terra + sum_{j=i}^{k-3} salt[j]  →  cumsum des de la dreta
        salt_cs = np.fliplr(np.cumsum(np.fliplr(salt_s), axis=1))  # (n_iter, k-2)
        w_cru[:, :k - 2] = (terra_s[:, None] + salt_cs)[:, inv_ordre]

    return w_cru / w_cru.sum(axis=1, keepdims=True)


def _generar_pesos_smaa(rng, n_iter, n_crit):
    """
    Mostreig uniforme sobre la regió factible w[0] > w[1] > ... > w[-2] = w[-1].
    Funciona per a n_crit >= 2; la darrera parella és sempre igual
    (estabilitat = promiscuïtat).

    Parametrització general (k = n_crit, k >= 3):
        u[j] = (j+1) * d[j]  per j = 0..k-3   (gaps entre criteris consecutius)
        u[k-2] = k * s                           (valor base dels dos últims)
        u ~ Dirichlet(1,...,1) de dimensió k-1  → sum(u) = 1
        d[j] = u[j] / (j+1),  s = u[k-2] / k
        w[k-1] = w[k-2] = s,  w[i] = s + sum(d[i..k-3])  per i < k-2
    Suma de pesos: k*s + sum_{j=0}^{k-3}(j+1)*d[j] = sum(u) = 1  ✓
    """
    k = n_crit
    if k == 1:
        return np.ones((n_iter, 1))
    if k == 2:
        w0 = rng.uniform(0.5, 1.0, size=n_iter)
        return np.column_stack([w0, 1.0 - w0])

    scales = np.array(list(range(1, k - 1)) + [k], dtype=float)   # (k-1,)
    u   = rng.dirichlet(np.ones(k - 1), size=n_iter)               # (n_iter, k-1)
    d_s = u / scales                                                 # (n_iter, k-1)

    s    = d_s[:, -1]    # (n_iter,)   — valor comú dels dos últims criteris
    gaps = d_s[:, :-1]   # (n_iter, k-2) — gaps d[0]..d[k-3]

    # w[i] = s + gaps[i] + gaps[i+1] + ... + gaps[k-3]
    gaps_cs = np.fliplr(np.cumsum(np.fliplr(gaps), axis=1))   # (n_iter, k-2)

    w = np.empty((n_iter, k))
    w[:, :k - 2] = s[:, None] + gaps_cs
    w[:, k - 2]  = s
    w[:, k - 1]  = s
    return w


# ─────────────────────────────────────────────
# SMAA VECTORITZAT
# ─────────────────────────────────────────────

def smaa_complet(subscores, te_dada, n_iter=200_000,
                 pesos_base=None, c=30.0,
                 mode="perturbacio", seed=None,
                 chunk_size=20_000):
    """
    Anàlisi SMAA vectoritzat per suportar moltes iteracions sense bloquejar.

    mode='perturbacio'  – pesos perturbats al voltant de pesos_base amb dispersió c%.
                        NO retorna central_weights (sempre None).
    mode='smaa'       – pesos Dirichlet uniformes (totalment aleatoris).
                        SÍ retorna central_weights per a cada candidat.

    Paràmetres
    ----------
    subscores  : (m, n_crit) float – sub-scores de cada OBP per a cada criteri
    te_dada    : (m, n_crit) bool  – True si la dada existeix; False → s'usa valor aleatori
    n_iter     : total d'iteracions de Monte Carlo
    pesos_base : llista de n_crit pesos (necessari per mode='perturbacio')
    c          : % de dispersió de la perturbació (mode='perturbacio')
    mode       : 'perturbacio' | 'smaa'
    seed       : llavor per a reproductibilitat
    chunk_size : iteracions per lot (controla l'ús de memòria)

    Retorna
    -------
    rank_accept  : (m, m) float – fracció de vegades que l'OBP i ocupa la posició j
    pairwise_dom : (m, m) float – fracció de vegades que l'OBP i supera l'OBP j
    central_w    : dict {i: array(n_crit)} o None
    """
    S = np.array(subscores, dtype=float)
    M = np.array(te_dada, dtype=bool)
    m, n_crit = S.shape
    rng = np.random.default_rng(seed)

    if mode not in ("perturbacio", "smaa"):
        raise ValueError("mode ha de ser 'perturbacio' o 'smaa'")
    if mode == "perturbacio" and pesos_base is None:
        raise ValueError("mode='perturbacio' necessita pesos_base")

    # Genera tots els pesos d'una vegada (forma: (n_iter, n_crit))
    if mode == "perturbacio":
        pesos_aleatoris = _generar_pesos_salts(rng, n_iter, pesos_base, c)
    else:
        pesos_aleatoris = _generar_pesos_smaa(rng, n_iter, n_crit)

    rank_counts  = np.zeros((m, m), dtype=np.int64)
    pairwise     = np.zeros((m, m), dtype=np.float64)
    central_acc  = [[] for _ in range(m)]   # acumula pesos guanyadors per a mode='smaa'

    # Màscares per difusió vectorial
    M3 = M[None, :, :]   # (1, m, n_crit) – per broadcasting amb chunk (n, m, n_crit)
    S3 = S[None, :, :]   # (1, m, n_crit)

    n_done = 0
    while n_done < n_iter:
        n = min(chunk_size, n_iter - n_done)
        w_chunk  = pesos_aleatoris[n_done:n_done + n]     # (n, n_crit)
        ale_chunk = rng.random((n, m, n_crit))             # (n, m, n_crit)

        # Substitueix criteris sense dada per valors aleatoris U(0,1)
        valors = np.where(M3, S3, ale_chunk)               # (n, m, n_crit)

        # Scores ponderats
        scores = (valors * w_chunk[:, None, :]).sum(axis=2)  # (n, m)

        # Rankings (posició 0 = millor)
        rankings = np.argsort(-scores, axis=1)               # (n, m)

        # Acumula rank_counts
        for pos in range(m):
            np.add.at(rank_counts, (rankings[:, pos], pos), 1)

        # Acumula central weights (mode='smaa', guanyadors)
        if mode == "smaa":
            winners = rankings[:, 0]                          # (n,) índex de l'OBP guanyador
            for i in range(m):
                mask = winners == i
                if mask.any():
                    central_acc[i].append(w_chunk[mask])

        # Dominació per parells: scores[k,i] > scores[k,j]
        pairwise += (scores[:, :, None] > scores[:, None, :]).sum(axis=0)

        n_done += n

    rank_accept  = rank_counts / n_iter
    pairwise_dom = pairwise / n_iter

    if mode == "smaa":
        central_w = {}
        for i in range(m):
            if central_acc[i]:
                central_w[i] = np.vstack(central_acc[i]).mean(axis=0)
            else:
                central_w[i] = None
    else:
        central_w = None

    return rank_accept, pairwise_dom, central_w
