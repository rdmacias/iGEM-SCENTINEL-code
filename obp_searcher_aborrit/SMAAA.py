import numpy as np


# ─────────────────────────────────────────────
# WEIGHT GENERATORS
# ─────────────────────────────────────────────

def _generate_jump_weights(rng, n_iter, base_weights, c):
    """
    Jump method: decompose into a floor (the value shared by the last two
    criteria) plus jumps (the gaps between consecutive criteria). Each piece
    moves uniformly within +/-c% of its base value; then it is rebuilt and
    normalized.

    c=0  -> fixed weights; c=100 -> each piece can double or vanish.

    Constraint: the last two criteria (stability = promiscuity) are forced to
    their mean and always stay equal (the jump between them is 0).
    The ordering w[0] >= w[1] >= ... >= w[-2] = w[-1] is required by
    construction - and since base_weights may come from the user (not
    necessarily sorted), the leading criteria (all but the last 2) are
    re-sorted from large to small and clipped to the floor before generating,
    and the output columns are put back in the original order so they line up
    with subscores/has_data. Without this, a leading weight below the floor
    gives inverted uniform(lo>hi) ranges - undefined in NumPy - and can
    produce negative weights.
    """
    base = np.array(base_weights, dtype=float)

    k = len(base)
    f = c / 100.0

    if k > 2:
        order     = np.argsort(-base[:-2], kind='stable')
        inv_order = np.argsort(order)
        base[:-2] = np.maximum(base[:-2][order], base[-1])

    # floor = value shared by the last two criteria
    floor = base[-1]
    # jumps[i] = base[i] - base[i+1] > 0, excluding the last pair (always 0)
    jumps = (-np.diff(base))[:-1]  # (k-2,)

    # The floor moves +/-c% (clipped to a minimum positive value)
    lo_t = max(floor * (1.0 - f), 1e-9)
    hi_t = floor * (1.0 + f)
    floor_s = rng.uniform(lo_t, hi_t, size=n_iter)  # (n_iter,)

    w_raw = np.empty((n_iter, k))
    w_raw[:, -2] = floor_s
    w_raw[:, -1] = floor_s

    if k > 2:
        # Each jump moves +/-c% (clipped at 0 from below - it cannot go negative)
        lo_jumps = np.maximum(jumps * (1.0 - f), 0.0)  # (k-2,)
        hi_jumps = jumps * (1.0 + f)                    # (k-2,)
        u = rng.random((n_iter, k - 2))
        jump_s = lo_jumps + u * (hi_jumps - lo_jumps)   # (n_iter, k-2)

        # w[:,i] = floor + sum_{j=i}^{k-3} jump[j]  ->  cumsum from the right
        jump_cs = np.fliplr(np.cumsum(np.fliplr(jump_s), axis=1))  # (n_iter, k-2)
        w_raw[:, :k - 2] = (floor_s[:, None] + jump_cs)[:, inv_order]

    return w_raw / w_raw.sum(axis=1, keepdims=True)


def _generate_smaa_weights(rng, n_iter, n_crit):
    """
    Uniform sampling over the feasible region w[0] > w[1] > ... > w[-2] = w[-1].
    Works for n_crit >= 2; the last pair is always equal
    (stability = promiscuity).

    General parametrization (k = n_crit, k >= 3):
        u[j] = (j+1) * d[j]  for j = 0..k-3   (gaps between consecutive criteria)
        u[k-2] = k * s                           (base value of the last two)
        u ~ Dirichlet(1,...,1) of dimension k-1  -> sum(u) = 1
        d[j] = u[j] / (j+1),  s = u[k-2] / k
        w[k-1] = w[k-2] = s,  w[i] = s + sum(d[i..k-3])  for i < k-2
    Sum of weights: k*s + sum_{j=0}^{k-3}(j+1)*d[j] = sum(u) = 1  OK
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

    s    = d_s[:, -1]    # (n_iter,)   - value shared by the last two criteria
    gaps = d_s[:, :-1]   # (n_iter, k-2) — gaps d[0]..d[k-3]

    # w[i] = s + gaps[i] + gaps[i+1] + ... + gaps[k-3]
    gaps_cs = np.fliplr(np.cumsum(np.fliplr(gaps), axis=1))   # (n_iter, k-2)

    w = np.empty((n_iter, k))
    w[:, :k - 2] = s[:, None] + gaps_cs
    w[:, k - 2]  = s
    w[:, k - 1]  = s
    return w


# ─────────────────────────────────────────────
# VECTORIZED SMAA
# ─────────────────────────────────────────────

def smaa_full(subscores, has_data, n_iter=200_000,
              base_weights=None, c=30.0,
              mode="perturbation", seed=None,
              chunk_size=20_000):
    """
    Vectorized SMAA analysis, so many iterations can run without blocking.

    mode='perturbation' - weights perturbed around base_weights with dispersion c%.
                        Does NOT return central_weights (always None).
    mode='smaa'         - uniform Dirichlet weights (fully random).
                        DOES return central_weights for each candidate.

    Parameters
    ----------
    subscores    : (m, n_crit) float - sub-scores of each OBP for each criterion
    has_data     : (m, n_crit) bool  - True if the value exists; False -> a random value is used
    n_iter       : total number of Monte Carlo iterations
    base_weights : list of n_crit weights (required for mode='perturbation')
    c            : % dispersion of the perturbation (mode='perturbation')
    mode         : 'perturbation' | 'smaa'
    seed         : seed for reproducibility
    chunk_size   : iterations per batch (controls memory usage)

    Returns
    -------
    rank_accept  : (m, m) float - fraction of times OBP i lands in position j
    pairwise_dom : (m, m) float - fraction of times OBP i beats OBP j
    central_w    : dict {i: array(n_crit)} or None
    """
    S = np.array(subscores, dtype=float)
    M = np.array(has_data, dtype=bool)
    m, n_crit = S.shape
    rng = np.random.default_rng(seed)

    if mode not in ("perturbation", "smaa"):
        raise ValueError("mode must be 'perturbation' or 'smaa'")
    if mode == "perturbation" and base_weights is None:
        raise ValueError("mode='perturbation' needs base_weights")

    # Generate every weight at once (shape: (n_iter, n_crit))
    if mode == "perturbation":
        random_weights = _generate_jump_weights(rng, n_iter, base_weights, c)
    else:
        random_weights = _generate_smaa_weights(rng, n_iter, n_crit)

    rank_counts  = np.zeros((m, m), dtype=np.int64)
    pairwise     = np.zeros((m, m), dtype=np.float64)
    central_acc  = [[] for _ in range(m)]   # accumulates winning weights for mode='smaa'

    # Masks for vector broadcasting
    M3 = M[None, :, :]   # (1, m, n_crit) - for broadcasting with chunk (n, m, n_crit)
    S3 = S[None, :, :]   # (1, m, n_crit)

    n_done = 0
    while n_done < n_iter:
        n = min(chunk_size, n_iter - n_done)
        w_chunk  = random_weights[n_done:n_done + n]     # (n, n_crit)
        rand_chunk = rng.random((n, m, n_crit))            # (n, m, n_crit)

        # Replace criteria without data by random U(0,1) values
        values = np.where(M3, S3, rand_chunk)              # (n, m, n_crit)

        # Weighted scores
        scores = (values * w_chunk[:, None, :]).sum(axis=2)  # (n, m)

        # Rankings (position 0 = best)
        rankings = np.argsort(-scores, axis=1)               # (n, m)

        # Accumulate rank_counts
        for pos in range(m):
            np.add.at(rank_counts, (rankings[:, pos], pos), 1)

        # Accumulate central weights (mode='smaa', winners)
        if mode == "smaa":
            winners = rankings[:, 0]                          # (n,) index of the winning OBP
            for i in range(m):
                mask = winners == i
                if mask.any():
                    central_acc[i].append(w_chunk[mask])

        # Pairwise dominance: scores[k,i] > scores[k,j]
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
