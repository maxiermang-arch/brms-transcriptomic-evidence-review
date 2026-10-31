"""Shared generator for original and seed-only module-reference experiments."""
import numpy as np
from scipy.stats import rankdata
from enhance_common import M, auc
from concordance_rules import right_tail

SCENARIOS = ['label_null', 'competitive_null', 'target_enriched']

def module_replicate(scenario, rep, seed_base=20261031):
    j = SCENARIOS.index(scenario)
    rng = np.random.default_rng(seed_base + 100000*j + rep)
    n, k, blocks = 52, 10, 40
    y = rng.permutation(np.repeat([False, True], n//2))
    target = int(rng.integers(blocks))
    effects = np.zeros(blocks) if j == 0 else rng.normal(0, .6, blocks)
    if j == 2:
        effects[target] += 1.5
    shared = rng.normal(size=(blocks, n))
    x = np.sqrt(.6)*np.repeat(shared, k, axis=0) + np.sqrt(.4)*rng.normal(size=(blocks*k, n))
    x += np.repeat(effects, k)[:, None]*y
    z = (x-x.mean(1)[:, None])/x.std(1, ddof=1)[:, None]
    modules = z.reshape(blocks, k, n).mean(1)
    score = modules[target]
    observed = auc(y, score)
    oracle = np.array([auc(y, v) for v in modules])
    po = float((oracle >= observed-1e-12).mean())
    sets = np.array([rng.choice(blocks*k, k, replace=False) for _ in range(399)])
    ru = rankdata(M.set_scores(z, sets), axis=1)
    pu = M.l1_pvalue(M.auc_from_ranks(ru, y), observed)
    _, _, rm = M.matched_reference(z, k, score.std(ddof=1), rng, rng, n_probe=20, n_null=399)
    pm = M.l1_pvalue(M.auc_from_ranks(rm, y), observed)
    ranks = rankdata(score)
    perms = np.array([rng.permutation(y) for _ in range(399)])
    perm_auc = (perms@ranks-y.sum()*(y.sum()+1)/2)/(y.sum()*(n-y.sum()))
    return dict(replicate=rep, scenario=scenario, auc=observed, p_oracle=po,
                p_size=pu, p_matched=pm, p_label=right_tail(observed, perm_auc))
