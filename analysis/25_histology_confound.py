"""analysis/25_histology_confound.py — R-3 组织学混淆检验

背景：模拟评审发现队列 A (GSE245467, n=52) 并非黑色素瘤队列，
      而是四组织学混合（LUN 17 / BRE 16 / MEL 14 / RCC 5），
      且 BrMS 标签与组织学强相关（chi2=48.59, p=1.98e-07, Cramers V=0.558）。

本脚本回答：
  1) 队列 A 的真实组织学构成 + histology × BrMS 交叉表 + 卡方/Cramers V
  2) 混淆强度：单用组织学能否预测 BrMS 标签？（histology-only AUC）
  3) 校正后：programme score 的效应是否存活？
     - 模型 A: programme score only
     - 模型 B: programme score + histology one-hot
     比较 score 的系数/OR/p 值在两模型间的变化
  4) 分层分析：在最大组织学层内（LUN n=17 / BRE n=16）分别算 programme AUC

输出: analysis/histology_confound_results.csv  + 控制台报告
      analysis/histology_only_auc.csv（含留一法主估计量与分离敏感性列）
"""
import os
import re
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import AN, SUBS, NM, BRMS_PROGRAMS, SEED

import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from scipy.stats import chi2_contingency

warnings_on = False
RNG = np.random.default_rng(SEED)

# Historical implementation uses Logit L1 regularisation, not ridge.
LOO_PENALTY = 0.25

def _loo_oof(Xh, y, penalty=0.0, method="newton"):
    """Held-out probabilities; positive penalty selects explicit L1 regularisation.
    No general invariance to penalty, optimiser or dummy coding is asserted.
    """
    n = len(y)
    oof = np.full(n, np.nan)
    mx = 0.0
    nfail = 0
    for i in range(n):
        tr = np.ones(n, bool)
        tr[i] = False
        if len(np.unique(y[tr])) < 2:
            continue
        try:
            if penalty > 0:
                m_i = sm.Logit(y[tr], Xh[tr]).fit_regularized(
                    alpha=penalty, method='l1', disp=0, maxiter=500)
                b = np.asarray(m_i.params, dtype=float)
                oof[i] = 1.0 / (1.0 + np.exp(-float(np.dot(Xh[i], b))))
            else:
                m_i = sm.Logit(y[tr], Xh[tr]).fit(disp=0, method=method)
                b = np.asarray(m_i.params, dtype=float)
                oof[i] = float(m_i.predict(Xh[i:i + 1])[0])
            mx = max(mx, float(np.max(np.abs(b))))
        except Exception:
            oof[i] = np.nan
            nfail += 1
    return oof, mx, nfail


def _auc_from_oof(y, oof):
    """由 out-of-fold 预测算 AUC；有效折不足或标签单一时返回 NaN。"""
    ok = np.isfinite(oof)
    if ok.sum() <= 5 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], oof[ok]))


def main():
    # ---------- 1) 标签 + 组织学 ----------
    lab = pd.read_parquet(f"{AN}/labels_used.parquet")
    lab["histology"] = [re.match(r"^([A-Z]+)", s).group(1) for s in lab.index]
    hist_map = {"LUNBRM": "Lung", "BREBRM": "Breast",
                "MELBRM": "Melanoma", "RCCBRM": "RCC"}
    lab["histo"] = lab["histology"].map(hist_map)

    print("=" * 78)
    print("R-3 组织学混淆检验")
    print("=" * 78)
    print("\n[1] 队列 A 组织学构成")
    print(lab["histo"].value_counts().to_string())
    print("\n[2] histology × BrMS 交叉表")
    ct = pd.crosstab(lab["histo"], lab["brms"])
    print(ct.to_string())
    chi2, p, dof, exp = chi2_contingency(ct.values)
    n = ct.values.sum()
    v = (chi2 / (n * (min(ct.shape) - 1))) ** 0.5
    print(f"\n  卡方 = {chi2:.2f}, dof = {dof}, p = {p:.3e}")
    print(f"  Cramers V = {v:.3f}")

    # ---------- 2) programme scores ----------
    Xc = pd.read_parquet(f"{AN}/X_counts.parquet")
    X = Xc[lab.index]
    cpm = X.div(X.sum(axis=0), axis=1) * 1e6
    L = np.log2(cpm + 1)
    ZA = L.sub(L.mean(axis=1), axis=0).div(L.std(axis=1).replace(0, np.nan), axis=0)
    REF = pd.DataFrame(index=lab.index)
    for sub in SUBS:
        gset = [g for g in BRMS_PROGRAMS[sub] if g in ZA.index]
        REF[sub] = ZA.loc[gset].mean(axis=0)
    REF_z = (REF - REF.mean(axis=0)) / REF.std(axis=0).replace(0, np.nan)

    # ---------- 3) 混淆强度：histology-only 预测 BrMS ----------
    print("\n[3] 混淆强度：仅用组织学 one-hot 预测 BrMS 标签 (one-vs-rest AUC)")
    H = pd.get_dummies(lab["histo"], prefix="h", drop_first=False).astype(float)
    conf_rows = []
    for sub in SUBS:
        y = (lab["brms"].astype(str) == sub).astype(int).values
        if y.sum() < 3:
            continue
        # one-vs-rest AUC using histology one-hot as score: 用 LR 拟合再取预测概率
        try:
            Xh = sm.add_constant(H.values, has_constant="add")
            res = sm.Logit(y, Xh).fit(disp=0)
            auc_h = roc_auc_score(y, res.predict(Xh))
        except Exception as e:
            auc_h = np.nan
        # 留一法交叉验证 AUC
        # 样本内 AUC 用 5 个参数（4 个组织学哑变量 + 截距）拟合 52 个样本，
        # 会偏乐观；判定规则 *Fail: AUC ≥ 0.80* 建立在"组织学能解释多少标签"
        # 上，因此必须给出样本外版本，并把样本内值降为描述性。
        # 2026-09-20：主估计量改为 L1 惩罚版，无惩罚版作敏感性。
        oof_r, mx_r, nf_r = _loo_oof(Xh, y, penalty=LOO_PENALTY)
        auc_loo = _auc_from_oof(y, oof_r)
        oof_u, mx_u, nf_u = _loo_oof(Xh, y, penalty=0.0, method="newton")
        auc_loo_unp = _auc_from_oof(y, oof_u)
        alt = {"bfgs": None, "lbfgs": None}
        for meth in alt:
            o, _, _ = _loo_oof(Xh, y, penalty=0.0, method=meth)
            alt[meth] = _auc_from_oof(y, o)
        vals = [v for v in [auc_loo_unp, *alt.values()] if np.isfinite(v)]
        spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
        sensitive = bool(spread > 1e-6)

        conf_rows.append({
            "subtype": sub, "n_pos": int(y.sum()),
            "histology_only_auc": round(auc_h, 3),
            # 主估计量：L1 α = 0.25
            "histology_only_auc_loo": round(float(auc_loo), 3),
            # 敏感性：无惩罚 MLE 在不同优化器下的取值
            "loo_unpenalized_newton": round(float(auc_loo_unp), 3),
            "loo_unpenalized_bfgs": round(float(alt["bfgs"]), 3),
            "loo_unpenalized_lbfgs": round(float(alt["lbfgs"]), 3),
            "loo_unpenalized_spread": round(float(spread), 3),
            "loo_separation_sensitive": sensitive,
            "loo_maxabs_beta_unpenalized": round(float(mx_u), 2),
            "loo_maxabs_beta_ridge": round(float(mx_r), 2),
        })
        print(f"  BrMS{sub}: n={int(y.sum())}, histology-only AUC = {auc_h:.3f} "
              f"(leave-one-out {auc_loo:.3f} [L1 α={LOO_PENALTY}], "
              f"无惩罚 {auc_loo_unp:.3f}/{alt['bfgs']:.3f}/{alt['lbfgs']:.3f}"
              f"{' ⚠️ 分离敏感' if sensitive else ''})")

    # ---------- 4) 校正后 programme score 效应 ----------
    print("\n[4] 组织学校正前后 programme score 的效应 (logistic, one-vs-rest)")
    print(f"  {'亚型':6s} {'模型':28s} {'score OR':>9s} {'95% CI':>22s} {'p':>10s} {'AUC':>7s}")
    rows = []
    Hv = H.values
    for sub in SUBS:
        y = (lab["brms"].astype(str) == sub).astype(int).values
        if y.sum() < 3:
            continue
        s = REF_z[sub].values

        # 模型 A: score only
        Xa = sm.add_constant(np.column_stack([s]), has_constant="add")
        ra = sm.Logit(y, Xa).fit(disp=0)
        or_a = np.exp(ra.params[1]); p_a = ra.pvalues[1]
        lo_a, hi_a = np.exp(ra.conf_int()[1])
        auc_a = roc_auc_score(y, ra.predict(Xa))

        # 模型 B: score + histology
        Xb = sm.add_constant(np.column_stack([s, Hv]), has_constant="add")
        rb = sm.Logit(y, Xb).fit(disp=0)
        or_b = np.exp(rb.params[1]); p_b = rb.pvalues[1]
        lo_b, hi_b = np.exp(rb.conf_int()[1])
        auc_b = roc_auc_score(y, rb.predict(Xb))

        print(f"  BrMS{sub:5s} {'A: score only':28s} {or_a:9.3f} "
              f"{f'[{lo_a:.2f}, {hi_a:.2f}]':>22s} {p_a:10.2e} {auc_a:7.3f}")
        print(f"  {'':6s} {'B: score + histology':28s} {or_b:9.3f} "
              f"{f'[{lo_b:.2f}, {hi_b:.2f}]':>22s} {p_b:10.2e} {auc_b:7.3f}")

        rows.append({
            "subtype": sub, "model": "score_only", "score_OR": round(or_a, 3),
            "ci_lo": round(lo_a, 3), "ci_hi": round(hi_a, 3),
            "p": float(f"{p_a:.3e}"), "auc": round(auc_a, 3),
        })
        rows.append({
            "subtype": sub, "model": "score_plus_histology", "score_OR": round(or_b, 3),
            "ci_lo": round(lo_b, 3), "ci_hi": round(hi_b, 3),
            "p": float(f"{p_b:.3e}"), "auc": round(auc_b, 3),
        })

    # ---------- 5) 分层分析 ----------
    print("\n[5] 分层分析：在单一组织学层内 programme AUC (one-vs-rest)")
    print(f"  {'层':10s} {'n':>4s} " + " ".join(f"{'BrMS'+s:>8s}" for s in SUBS))
    for h in ["Lung", "Breast", "Melanoma"]:
        m = (lab["histo"] == h).values
        if m.sum() < 8:
            print(f"  {h:10s} {m.sum():4d}  (n<8, 跳过)")
            continue
        vals = []
        for sub in SUBS:
            y = (lab.loc[m, "brms"].astype(str) == sub).astype(int).values
            if y.sum() < 2 or len(np.unique(y)) < 2:
                vals.append("   n/a  ")
            else:
                vals.append(f"{roc_auc_score(y, REF_z.loc[m, sub].values):8.3f}")
        print(f"  {h:10s} {m.sum():4d} " + " ".join(vals))

    # ---------- 保存 ----------
    out = pd.DataFrame(rows)
    out.to_csv(f"{AN}/histology_confound_results.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(f"{AN}/histology_only_auc.csv", index=False)
    print(f"\n  ✅ Wrote {AN}/histology_confound_results.csv")
    print(f"  ✅ Wrote {AN}/histology_only_auc.csv")


if __name__ == "__main__":
    main()
