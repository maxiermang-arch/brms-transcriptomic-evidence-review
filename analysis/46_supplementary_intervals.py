"""analysis/46_supplementary_intervals.py — 区间与阈值稳健性（第二轮审稿 T10 / T11 / T12）

三个区块，各自只回答一个"这个数字有多不确定"的问题：

  A. **PDO organoid 区间**（支撑 §3.9 与 Supplementary Table S6 / T10）
     每组 n_in / n_out、两组中位 IC50、差值、比值、rank-biserial 的 bootstrap 区间。
     目的：暴露 n_in = 2/3 的组（gamitrinib BrMS1–BrMS3）其点估计不可解释。

  B. **代谢组 PCA AUC 区间**（支撑 §3.10 与 Supplementary Methods SM2 表 / T12）
     PCA 分离统计量的 bootstrap 区间 + 置换 *p* 的**分辨率下限**。
     目的：BrMS2 的 0.694 来自 2 个样本、BrMS1 的 0.923 来自 7 个；
          以及 BrMS1 报出的 *p* = 0.001 恰好是 1,000 次置换的下限。

  C. **组织学混淆阈值稳健性**（支撑 §4.x 与 T11）
     histology-only AUC 的样本内 / 留一（LOO）两口径、LOO 的 bootstrap 区间、
     判定阈值 0.60–0.95 网格上被标记的亚型集合、以及单一组织学层内的 programme AUC。

口径纪律：三块的装载与统计量**逐字复用**上游脚本，不引入新自由度。
  - A 直接 import `analysis/23_pdo_ic50_test.py` 的 `load_pdo()`
  - B 复刻 `analysis/10_gold_standard_collision.py` 的 Data 1→15 映射与 Z 矩阵、PCA 统计量
  - C 复刻 `analysis/25_histology_confound.py` 的 histo 映射、REF_z 与 LOO 循环，
      并对历史产物 `histology_only_auc.csv` 做逐位复现校验

输出（全部覆盖式重写，幂等）：
  analysis/pdo_ic50_intervals.csv
  analysis/metabolomics_auc_intervals.csv
  analysis/histology_threshold_robustness.csv
  analysis/histology_stratified_auc.csv

RNG：全部固定到 `_common.SEED`（20260913），与 26_/32_/44_/45_ 同一约定。
"""
from __future__ import annotations

import importlib
import os
import re
import sys
import warnings
from math import comb

import numpy as np
import openpyxl
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import AN, ROOT, SEED, SUBS, NM, BRMS_PROGRAMS  # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- 常量
N_BOOT = 5000            # A 区：每组样本极小，bootstrap 可以给足
N_BOOT_PCA = 2000        # B 区：每次重复要重拟合 PCA，取 2,000
N_BOOT_C = 2000          # C 区
N_PERM_METAB = 1000      # 与 10_ 脚本一致（用于还原 p 的分辨率下限）
CI = 95.0

RNG = np.random.default_rng(SEED)


# ---------------------------------------------------------------- 工具
def pct_ci(x: np.ndarray) -> tuple[float, float]:
    """百分位 bootstrap 区间（丢弃 NaN）。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    lo, hi = np.percentile(x, [(100 - CI) / 2, 100 - (100 - CI) / 2])
    return float(lo), float(hi)


def fmt(v: float, nd: int = 3) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:.{nd}f}"


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == y.size:
        return np.nan
    return float(roc_auc_score(y, s))


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """1 − 2U/(n1·n0)，与 23_ 脚本同一约定（>0 表示 a 组秩更高）。"""
    n1, n0 = len(a), len(b)
    if n1 == 0 or n0 == 0:
        return np.nan
    allv = np.concatenate([a, b])
    r = pd.Series(allv).rank().values
    U = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(1 - 2 * U / (n1 * n0))


# ================================================================ A. PDO
def part_a() -> pd.DataFrame:
    print("=" * 90)
    print("A. PDO organoid：每组样本数与中位 IC50 的 bootstrap 区间")
    print("=" * 90)
    m23 = importlib.import_module("23_pdo_ic50_test")
    pdo = m23.load_pdo()
    drugs = ["abemaciclib", "everolimus", "gamitrinib"]
    preset = {"4": "abemaciclib", "3": "gamitrinib"}

    print(f"  PDO 例数 n={len(pdo)}；"
          f"BrMS 分布 {pdo['brms'].value_counts().sort_index().to_dict()}")
    print(f"  可用 IC50："
          + " ".join(f"{d} n={int(pdo[d].notna().sum())}" for d in drugs))

    rows = []
    for drug in drugs:
        for sub in SUBS:
            v = pdo[drug].dropna()
            lab = pdo.loc[v.index, "brms"]
            a = v[lab == sub].values.astype(float)
            b = v[lab != sub].values.astype(float)
            if len(a) < 2 or len(b) < 2:
                continue

            med_a, med_b = float(np.median(a)), float(np.median(b))
            diff = med_a - med_b
            ratio = med_a / med_b if med_b != 0 else np.nan
            rb = rank_biserial(a, b)

            ba = np.empty(N_BOOT); bb = np.empty(N_BOOT)
            for i in range(N_BOOT):
                sa = a[RNG.integers(0, len(a), len(a))]
                sb = b[RNG.integers(0, len(b), len(b))]
                ba[i] = np.median(sa)
                bb[i] = np.median(sb)
            bd, br = ba - bb, np.divide(ba, bb, out=np.full(N_BOOT, np.nan), where=bb != 0)
            # P(中位 IC50 在亚型组更低) —— 不含尺度假设的方向一致性度量
            p_lower = float(np.mean(ba < bb)) if np.isfinite(bd).any() else np.nan

            da_lo, da_hi = pct_ci(ba)
            db_lo, db_hi = pct_ci(bb)
            dd_lo, dd_hi = pct_ci(bd)
            dr_lo, dr_hi = pct_ci(br)

            rows.append({
                "drug": drug, "subtype": sub,
                "preset_direction": preset.get(sub) == drug,
                "n_in": len(a), "n_out": len(b),
                "median_in": round(med_a, 4), "median_in_lo": round(da_lo, 4),
                "median_in_hi": round(da_hi, 4),
                "median_out": round(med_b, 4), "median_out_lo": round(db_lo, 4),
                "median_out_hi": round(db_hi, 4),
                "median_diff": round(diff, 4),
                "median_diff_lo": round(dd_lo, 4), "median_diff_hi": round(dd_hi, 4),
                "ratio_in_over_out": round(ratio, 4) if np.isfinite(ratio) else np.nan,
                "ratio_lo": round(dr_lo, 4), "ratio_hi": round(dr_hi, 4),
                "rank_biserial": round(rb, 4),
                "boot_p_in_lower": round(p_lower, 4),
                "n_boot": N_BOOT,
            })

    out = pd.DataFrame(rows)
    out.to_csv(f"{AN}/pdo_ic50_intervals.csv", index=False)

    print(f"\n  {'药':13s} {'亚型':4s} {'n':>7s} {'中位(亚型)':>22s} "
          f"{'中位(其余)':>22s} {'比值':>18s} {'P(更低)':>8s}")
    for _, r in out.iterrows():
        tag = " *" if r["preset_direction"] else "  "
        print(f"  {r['drug']:11s}{tag}{r['subtype']:4s} "
              f"{r['n_in']:2d} vs {r['n_out']:2d} "
              f"{fmt(r['median_in'])+' ['+fmt(r['median_in_lo'])+', '+fmt(r['median_in_hi'])+']':>22s} "
              f"{fmt(r['median_out'])+' ['+fmt(r['median_out_lo'])+', '+fmt(r['median_out_hi'])+']':>22s} "
              f"{fmt(r['ratio_in_over_out'])+' ['+fmt(r['ratio_lo'])+', '+fmt(r['ratio_hi'])+']':>18s} "
              f"{r['boot_p_in_lower']:8.3f}")

    # 与 45_ 的功效曲线挂钩：点估计不显著究竟是"真无差异"还是"功效不足"
    er_path = f"{AN}/error_rates.csv"
    if os.path.exists(er_path):
        er = pd.read_csv(er_path)
        l5 = er[er["rule"].astype(str).str.startswith("L5_")]
        print("\n  ── 功效背景（来自 45_error_rates.py，本脚本不重算）")
        for _, r in l5.iterrows():
            print(f"     L5 {r['cohort']:26s} {r['scenario']:16s} "
                  f"通过率 {r['pass_rate']:.3f}")
    print(f"\n  ✅ Wrote {AN}/pdo_ic50_intervals.csv ({len(out)} rows)")
    return out


# ================================================================ B. 代谢组
def _load_metabolomics() -> tuple[pd.DataFrame, pd.Series]:
    """逐字复刻 10_gold_standard_collision.py 的 Data 1 → Data 15 映射。"""
    XL = f"{ROOT}/data/supplementary/41467_2026_68748_MOESM3_ESM.xlsx"
    wb = openpyxl.load_workbook(XL, read_only=True)

    def sheet(sn, header_row=1):
        ws = wb[sn]
        rows = list(ws.iter_rows(values_only=True))
        head = [str(x).strip() if x is not None else f"c{i}"
                for i, x in enumerate(rows[header_row])]
        return pd.DataFrame(rows[header_row + 1:], columns=head)

    d1 = sheet("Supplementary Data 1", 1)
    d1 = d1[d1["Sample_ID"].notna()]
    idmap, hit = {}, []
    for _, r in d1.iterrows():
        bb = str(r["BrMS"]).strip()
        if bb in SUBS:
            idmap[str(r["Sample_ID"]).strip()] = {"brms": bb}
    d15 = sheet("Supplementary Data 15", 1)
    d15 = d15.rename(columns={d15.columns[0]: "metabolite"})
    metab = d15.set_index("metabolite")
    metab.columns = [str(c).strip() for c in metab.columns]
    metab = metab.apply(pd.to_numeric, errors="coerce")
    hit = [c for c in metab.columns if c in idmap]
    for key in ["Patient ID", "Sample_ID"]:
        m2 = {}
        for _, r in d1.iterrows():
            bb = str(r["BrMS"]).strip()
            if bb in SUBS and pd.notna(r[key]):
                m2[str(r[key]).strip()] = bb
        hit2 = [c for c in metab.columns if c in m2]
        if len(hit2) > len(hit):
            hit = hit2
            idmap = {k: {"brms": v} for k, v in m2.items()}

    X = metab[hit]
    lab = pd.Series({c: idmap[c]["brms"] for c in hit})
    X = X[X.isna().mean(axis=1) <= 0.3]
    X = X.fillna(X.median(axis=1), axis=0)
    Z = X.sub(X.mean(axis=1), axis=0).div(
        X.std(axis=1).replace(0, np.nan), axis=0).dropna(how="all")
    return Z, lab


def _pca_best_auc(Z: pd.DataFrame, y: np.ndarray) -> float:
    """与 10_ 脚本完全相同的统计量：3 个主成分各自折到 ≥0.5 后取最大。"""
    sc = PCA(n_components=3, random_state=0).fit_transform(Z.T.values)
    return float(max(max(roc_auc_score(y, sc[:, i]), 1 - roc_auc_score(y, sc[:, i]))
                     for i in range(3)))


def part_b() -> pd.DataFrame:
    print("\n" + "=" * 90)
    print("B. 代谢组 PCA one-vs-rest AUC 的 bootstrap 区间与置换 p 的分辨率下限")
    print("=" * 90)
    Z, lab = _load_metabolomics()
    n_s, k_meta = Z.shape[1], Z.shape[0]
    print(f"  可用代谢物 {k_meta} × 样本 {n_s}；亚型分布 "
          f"{dict(lab.value_counts().sort_index())}")

    rows = []
    for sub in SUBS:
        y = (lab == sub).astype(int).values
        k = int(y.sum())
        obs = _pca_best_auc(Z, y)
        boot = np.full(N_BOOT_PCA, np.nan)
        for i in range(N_BOOT_PCA):
            idx = RNG.integers(0, n_s, n_s)
            yb = y[idx]
            if yb.sum() == 0 or yb.sum() == yb.size:
                continue
            boot[i] = _pca_best_auc(Z.iloc[:, idx], yb)
        ok = np.isfinite(boot)
        lo, hi = pct_ci(boot)
        # 置换 s 的分辨率下限：1,000 次置换 → 最小可报 p = 1/1001
        p_floor_rand = 1.0 / (N_PERM_METAB + 1)
        n_distinct = comb(n_s, k)
        rows.append({
            "subtype": sub, "subtype_name": NM[sub],
            "n_pos": k, "n_neg": n_s - k, "n_samples": n_s, "n_metabolites": k_meta,
            "PCA_best_auc": round(obs, 4),
            "boot_median": round(float(np.nanmedian(boot)), 4),
            "auc_lo": round(lo, 4), "auc_hi": round(hi, 4),
            "n_boot_valid": int(ok.sum()), "n_boot": N_BOOT_PCA,
            "auc_ci_width": round(hi - lo, 4),
            "n_distinct_labellings": n_distinct,
            "perm_p_floor_random": round(p_floor_rand, 6),
        })

    out = pd.DataFrame(rows)
    out.to_csv(f"{AN}/metabolomics_auc_intervals.csv", index=False)

    # 历史 p 值的下限判定（读 10_ 脚本的产物，不重跑置换）
    hist = pd.read_csv(f"{AN}/metabolomics_vs_BrMS.csv")
    pmap = {str(r["subtype"]): float(r["PCA_perm_p"]) for _, r in hist.iterrows()}
    nmaplab = {NM[s]: s for s in SUBS}

    print(f"\n  {'亚型':22s} {'n+':>3s} {'AUC':>6s} {'95% bootstrap CI':>20s} "
          f"{'CI宽':>7s} {'置换p':>8s}  下限?")
    out["PCA_perm_p"] = np.nan
    out["perm_p_at_floor"] = False
    # 上游 metabolomics_vs_BrMS.csv 把 p 四舍五入到 4 位小数，故下限比较必须留半个
    # 舍入单位的容差，否则 p = 0.001（真值 1/1001 = 0.000999…）会被判为"未触底"。
    tol = 5e-5
    for _, r in out.iterrows():
        lbl = r["subtype_name"]
        p = pmap.get(lbl, np.nan)
        at_floor = bool(np.isfinite(p) and p - r["perm_p_floor_random"] <= tol)
        out.loc[out["subtype"] == r["subtype"], "PCA_perm_p"] = p
        out.loc[out["subtype"] == r["subtype"], "perm_p_at_floor"] = at_floor
        print(f"  {lbl:22s} {r['n_pos']:3d} {r['PCA_best_auc']:6.3f} "
              f"{'['+fmt(r['auc_lo'])+', '+fmt(r['auc_hi'])+']':>20s} "
              f"{r['auc_ci_width']:7.3f} {fmt(p):>8s}  {'★是（下限）' if at_floor else ''}")
    out.to_csv(f"{AN}/metabolomics_auc_intervals.csv", index=False)

    print(f"\n  说明：置换 p 由 {N_PERM_METAB} 次置换产生 → 最小可报值 "
          f"{1/(N_PERM_METAB+1):.4f}；标 ★ 者等于该下限，"
          f"真实 p 只知 '低于下限'，不可当作精确值。")
    print(f"  说明：BrMS2 的 n={int(out.loc[out['subtype']=='2','n_pos'].iloc[0])} "
          f"→ 其 AUC 与区间不构成检验。")
    print(f"\n  ✅ Wrote {AN}/metabolomics_auc_intervals.csv ({len(out)} rows)")
    return out


# ================================================================ C. 组织学
def _histo_labels() -> pd.DataFrame:
    lab = pd.read_parquet(f"{AN}/labels_used.parquet")
    lab["histology"] = [re.match(r"^([A-Z]+)", s).group(1) for s in lab.index]
    hmap = {"LUNBRM": "Lung", "BREBRM": "Breast", "MELBRM": "Melanoma", "RCCBRM": "RCC"}
    lab["histo"] = lab["histology"].map(hmap)
    return lab


def _programme_scores(lab: pd.DataFrame) -> pd.DataFrame:
    """逐字复刻 25_histology_confound.py 的 REF_z。"""
    Xc = pd.read_parquet(f"{AN}/X_counts.parquet")
    X = Xc[lab.index]
    cpm = X.div(X.sum(axis=0), axis=1) * 1e6
    L = np.log2(cpm + 1)
    ZA = L.sub(L.mean(axis=1), axis=0).div(L.std(axis=1).replace(0, np.nan), axis=0)
    REF = pd.DataFrame(index=lab.index)
    for sub in SUBS:
        gset = [g for g in BRMS_PROGRAMS[sub] if g in ZA.index]
        REF[sub] = ZA.loc[gset].mean(axis=0)
    return (REF - REF.mean(axis=0)) / REF.std(axis=0).replace(0, np.nan)


def part_c() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 90)
    print("C. 组织学混淆：LOO 估计、bootstrap 区间、判定阈值网格与分层分析")
    print("=" * 90)
    import statsmodels.api as sm
    m25 = importlib.import_module("25_histology_confound")

    lab = _histo_labels()
    REFz = _programme_scores(lab)
    H = pd.get_dummies(lab["histo"], prefix="h", drop_first=False).astype(float)
    Hv = H.values
    n = len(lab)

    print(f"  样本 n={n}；组织学构成 "
          f"{dict(lab['histo'].value_counts())}")
    print(f"  留一法主估计量：L2 惩罚 α={m25.LOO_PENALTY}（复用 25_ 的 _loo_oof，"
          f"解唯一）")

    rows, oof_store = [], {}
    for sub in SUBS:
        y = (lab["brms"].astype(str) == sub).astype(int).values
        if y.sum() < 3:
            continue
        Xh = sm.add_constant(Hv, has_constant="add")
        # in-sample（仅描述性）
        try:
            m = sm.Logit(y, Xh).fit(disp=0)
            auc_in = float(roc_auc_score(y, m.predict(Xh)))
        except Exception:
            auc_in = np.nan
        # leave-one-out：主估计量 = ridge，敏感性 = 无惩罚的三个优化器
        oof_r, mx_r, nf_r = m25._loo_oof(Xh, y, penalty=m25.LOO_PENALTY)
        auc_loo = m25._auc_from_oof(y, oof_r)
        oof_u, mx_u, nf_u = m25._loo_oof(Xh, y, penalty=0.0, method="newton")
        auc_unp = m25._auc_from_oof(y, oof_u)
        alt = {}
        for meth in ("bfgs", "lbfgs"):
            o, _, _ = m25._loo_oof(Xh, y, penalty=0.0, method=meth)
            alt[meth] = m25._auc_from_oof(y, o)
        vals = [v for v in [auc_unp, *alt.values()] if np.isfinite(v)]
        spread = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
        oof_store[sub] = (y, oof_r)

        # LOO AUC 的 bootstrap：重抽样本及其 out-of-fold 预测（估计量已固定）
        boot = np.full(N_BOOT_C, np.nan)
        ok = np.isfinite(oof_r)
        yv, ov = y[ok], oof_r[ok]
        for i in range(N_BOOT_C):
            j = RNG.integers(0, len(yv), len(yv))
            boot[i] = safe_auc(yv[j], ov[j])
        lo, hi = pct_ci(boot)
        rows.append({
            "subtype": sub, "subtype_name": NM[sub], "n_pos": int(y.sum()),
            "histology_only_auc_in_sample": round(auc_in, 4),
            "histology_only_auc_loo": round(auc_loo, 4),
            "loo_unpenalized_newton": round(float(auc_unp), 4),
            "loo_unpenalized_bfgs": round(float(alt["bfgs"]), 4),
            "loo_unpenalized_lbfgs": round(float(alt["lbfgs"]), 4),
            "loo_unpenalized_spread": round(float(spread), 4),
            "loo_separation_sensitive": bool(spread > 1e-6),
            "loo_maxabs_beta_unpenalized": round(float(mx_u), 2),
            "loo_maxabs_beta_ridge": round(float(mx_r), 2),
            "loo_penalty_alpha": m25.LOO_PENALTY,
            "loo_lo": round(lo, 4), "loo_hi": round(hi, 4),
            "loo_ci_width": round(hi - lo, 4),
            "n_boot_valid": int(np.isfinite(boot).sum()), "n_boot": N_BOOT_C,
        })

    res = pd.DataFrame(rows)

    # ---- 与 25_ 的产物逐位比对（口径漂移检测）----
    old_path = f"{AN}/histology_only_auc.csv"
    drift = []
    if os.path.exists(old_path):
        old = pd.read_csv(old_path).set_index("subtype")
        for _, r in res.iterrows():
            s = int(r["subtype"])
            if s in old.index:
                a0 = float(old.loc[s, "histology_only_auc"])
                b0 = float(old.loc[s, "histology_only_auc_loo"])
                if abs(a0 - r["histology_only_auc_in_sample"]) > 5e-4 or \
                   abs(b0 - r["histology_only_auc_loo"]) > 5e-4:
                    drift.append((s, a0, r["histology_only_auc_in_sample"],
                                  b0, r["histology_only_auc_loo"]))
    if drift:
        print("  ⚠️ 与 histology_only_auc.csv 不一致（口径漂移）：")
        for s, a0, a1, b0, b1 in drift:
            print(f"     BrMS{s}: in-sample {a0:.3f}→{a1:.3f} | LOO {b0:.3f}→{b1:.3f}")
    else:
        print("  ✅ 复现校验：in-sample 与 LOO(ridge) 两列与 histology_only_auc.csv 逐位一致")
        print("     （2026-09-20 起 25_ 的 LOO 主估计量已改为 ridge α=0.25，"
              "无惩罚值作敏感性另列）")

    print(f"\n  {'亚型':22s} {'n+':>3s} {'样本内':>8s} {'LOO(ridge)':>11s} "
          f"{'LOO 95% bootstrap CI':>22s} {'CI宽':>7s} {'无惩罚区间':>14s}")
    for _, r in res.iterrows():
        s = ""
        if r["loo_separation_sensitive"]:
            s = " ⚠️"
        unp = (f"{r['loo_unpenalized_newton']:.3f}–"
               f"{max(r['loo_unpenalized_newton'], r['loo_unpenalized_bfgs'], r['loo_unpenalized_lbfgs']):.3f}")
        print(f"  {r['subtype_name']:22s} {r['n_pos']:3d} "
              f"{r['histology_only_auc_in_sample']:8.3f} "
              f"{r['histology_only_auc_loo']:11.3f} "
              f"{'['+fmt(r['loo_lo'])+', '+fmt(r['loo_hi'])+']':>22s} "
              f"{r['loo_ci_width']:7.3f} {unp:>14s}{s}")

    res.to_csv(f"{AN}/histology_only_auc_with_ci.csv", index=False)

    # ---- 阈值稳健性网格 ----
    grid = np.round(np.arange(0.60, 0.951, 0.05), 2)
    trows = []
    for tau in grid:
        for col, est in [("histology_only_auc_in_sample", "in_sample"),
                         ("histology_only_auc_loo", "loo")]:
            flagged = [int(r["subtype"]) for _, r in res.iterrows()
                       if np.isfinite(r[col]) and r[col] >= tau]
            trows.append({
                "threshold": float(tau), "estimand": est,
                "n_flagged": len(flagged),
                "flagged": "+".join(f"BrMS{s}" for s in flagged) if flagged else "(none)",
            })
    thr = pd.DataFrame(trows)
    thr.to_csv(f"{AN}/histology_threshold_robustness.csv", index=False)

    print(f"\n  阈值稳健性（判定规则：histology-only AUC ≥ τ 记为 evidence of failure）")
    print(f"  {'τ':>5s}  {'样本外(LOO)':38s}  样本内(仅描述性)")
    for tau in grid:
        g = thr[thr["threshold"] == tau].set_index("estimand")
        print(f"  {tau:5.2f}  {g.loc['loo','flagged']:38s}  "
              f"{g.loc['in_sample','flagged']}")
    lo_set = set(thr[(thr["estimand"] == "loo") & (thr["threshold"].between(0.70, 0.85))]["flagged"])
    in_sets = set(thr[thr["estimand"] == "in_sample"]["flagged"])
    print(f"\n  LOO 口径在 τ ∈ [0.70, 0.85] 上标记集合唯一：{sorted(lo_set)}")
    print(f"  样本内口径出现 {len(in_sets)} 种不同标记集合 → "
          f"判定随阈值漂移（这是把样本内值降为描述性的又一理由）")

    # ---- 分层分析（单一组织学层内）----
    # `auc` 是**有正例数下限**的版本（n_pos < 2 记 NaN，不可估计）；
    # `auc_unguarded` 复刻 34_support_stats.py 的原始口径（只要有正例就算，
    # 哪怕 n_pos = 1），用于与 stratified_auc.csv 逐位对账。
    srows = []
    for h in ["Lung", "Breast", "Melanoma", "RCC"]:
        m = (lab["histo"] == h).values
        for sub in SUBS:
            y = (lab.loc[m, "brms"].astype(str) == sub).astype(int).values
            s = REFz.loc[m, sub].values
            yb = y
            small = f"small stratum (n = {int(m.sum())}), descriptive only" if m.sum() < 8 else ""
            if yb.sum() == 0 or yb.sum() == yb.size:
                srows.append({"histology": h, "subtype": sub, "n_stratum": int(m.sum()),
                              "n_pos": int(yb.sum()), "auc": np.nan, "auc_unguarded": np.nan,
                              "auc_lo": np.nan, "auc_hi": np.nan, "estimable": False,
                              "note": "no positive sample"})
                continue
            auc_ung = safe_auc(yb, s)
            if yb.sum() < 2 or yb.sum() > yb.size - 2:
                srows.append({"histology": h, "subtype": sub, "n_stratum": int(m.sum()),
                              "n_pos": int(yb.sum()), "auc": np.nan,
                              "auc_unguarded": round(auc_ung, 4),
                              "auc_lo": np.nan, "auc_hi": np.nan, "estimable": False,
                              "note": ("not estimable (fewer than 2 samples on one side)"
                                       + ("; " + small if small else ""))})
                continue
            auc = safe_auc(yb, s)
            boot = np.full(N_BOOT_C, np.nan)
            for i in range(N_BOOT_C):
                j = RNG.integers(0, len(yb), len(yb))
                boot[i] = safe_auc(yb[j], s[j])
            lo, hi = pct_ci(boot)
            note = "reversed direction (AUC < 0.5)" if np.isfinite(auc) and auc < 0.5 else ""
            if small:
                note = (note + "; " if note else "") + small
            srows.append({"histology": h, "subtype": sub, "n_stratum": int(m.sum()),
                          "n_pos": int(yb.sum()), "auc": round(auc, 4),
                          "auc_unguarded": round(auc_ung, 4),
                          "auc_lo": round(lo, 4), "auc_hi": round(hi, 4),
                          "estimable": True, "note": note})
    st = pd.DataFrame(srows)

    # ---- 与 34_ 的 raw 表逐位对账 ----
    raw_path = f"{AN}/stratified_auc.csv"
    if os.path.exists(raw_path):
        raw = pd.read_csv(raw_path, dtype={"subtype": str})
        bad = []
        for _, r in st.iterrows():
            q = raw[(raw["stratum"] == r["histology"]) & (raw["subtype"] == r["subtype"])]
            if q.empty:
                continue
            v0 = q.iloc[0]["auc"]
            v1 = r["auc_unguarded"]
            if pd.isna(v0) and pd.isna(v1):
                continue
            if pd.isna(v0) != pd.isna(v1) or abs(float(v0) - float(v1)) > 5e-4:
                bad.append((r["histology"], r["subtype"], v0, v1))
        print("  " + ("✅ 对账：auc_unguarded 与 34_support_stats.py 的 "
                      "stratified_auc.csv 逐位一致"
                      if not bad else f"⚠️ 与 stratified_auc.csv 不一致：{bad}"))

    st.to_csv(f"{AN}/histology_stratified_auc.csv", index=False)

    print(f"\n  分层分析：单一组织学层内的 programme one-vs-rest AUC")
    print(f"  ⚠️ 此 AUC **不折叠方向**：< 0.5 表示该程序分数在该层内反向关联")
    print(f"  {'层':10s} {'n':>4s} {'亚型':6s} {'n+':>3s} {'AUC':>6s} "
          f"{'95% CI':>18s}  备注")
    for _, r in st.iterrows():
        ci = "n/a" if not np.isfinite(r["auc_lo"]) else f"[{r['auc_lo']:.3f}, {r['auc_hi']:.3f}]"
        note = r["note"]
        if not note and np.isfinite(r["auc"]) and r["auc"] < 0.5:
            note = "⚠️ 反向"
        print(f"  {r['histology']:10s} {r['n_stratum']:4d} BrMS{r['subtype']:3s} "
              f"{r['n_pos']:3d} {fmt(r['auc']):>6s} {ci:>18s}  {note}")

    print(f"\n  ✅ Wrote {AN}/histology_only_auc_with_ci.csv ({len(res)} rows)")
    print(f"  ✅ Wrote {AN}/histology_threshold_robustness.csv ({len(thr)} rows)")
    print(f"  ✅ Wrote {AN}/histology_stratified_auc.csv ({len(st)} rows)")
    return res, st


def main():
    print("=" * 90)
    print("46_supplementary_intervals.py — 区间与阈值稳健性（T10 / T11 / T12）")
    print(f"N_BOOT(A)={N_BOOT}  N_BOOT(B)={N_BOOT_PCA}  N_BOOT(C)={N_BOOT_C}  "
          f"SEED={SEED}")
    print("=" * 90)
    part_a()
    part_b()
    part_c()
    print("\n" + "=" * 90)
    print("完成。四个产物：pdo_ic50_intervals.csv / metabolomics_auc_intervals.csv")
    print("                  histology_only_auc_with_ci.csv / histology_threshold_robustness.csv")
    print("                  histology_stratified_auc.csv")
    print("=" * 90)


if __name__ == "__main__":
    main()
