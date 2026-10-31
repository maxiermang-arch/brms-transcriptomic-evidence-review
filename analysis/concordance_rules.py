"""Positive concordance with an explicit eight-test family and aggregation."""
import numpy as np
from scipy.stats import rankdata

def right_tail(observed, null):
    null = np.asarray(null)
    return (1 + np.count_nonzero(null >= observed - 1e-12)) / (len(null) + 1)

def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum(1, np.minimum.accumulate((p[order] * p.size / np.arange(1, p.size + 1))[::-1])[::-1])
    return q

def aggregate(rho, p):
    rho = np.asarray(rho).reshape(2, 4)
    q = bh(np.asarray(p).ravel()).reshape(2, 4)
    supported = (rho > 0) & (q <= .05)
    return supported.all(0), supported.any(0), q

def screen(a, b, ya, yb, rng, nperm=2000):
    """Rows are within-signature sample ranks; fixed A, permuted B labels."""
    classes = np.unique(ya)
    perms = np.array([rng.permutation(yb) for _ in range(nperm)])
    rho, p = [], []
    def aucs(r, y):
        n1 = y.sum(-1); n0 = r.shape[1] - n1
        return (y @ r.T - (n1 * (n1 + 1) / 2)[..., None]) / (n1 * n0)[..., None]
    for s in classes:
        av = aucs(a, (ya == s).astype(float)); bv = aucs(b, (yb == s).astype(float))
        ar = rankdata(av); ar -= ar.mean()
        br = rankdata(bv); br -= br.mean()
        den = np.linalg.norm(ar) * np.linalg.norm(br)
        obs = float(ar @ br / den) if den else 0.
        ranks = rankdata(aucs(b, (perms == s).astype(float)), axis=1)
        ranks -= ranks.mean(1)[:, None]
        denom = np.linalg.norm(ranks, axis=1) * np.linalg.norm(ar)
        null = np.divide(ranks @ ar, denom, out=np.zeros(nperm), where=denom > 0)
        rho.append(obs); p.append(right_tail(obs, null))
    return rho, p
