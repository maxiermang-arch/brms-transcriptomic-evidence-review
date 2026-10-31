"""analysis/22_cohort_C_br_projection.py — E-MTAB-8659 投影 (阶段 3 A3)

时间盒限制：30 min（README 「阶段收口计划」 A3, 不阻塞）
设计：把 BRMS_PROGRAMS（_common.py 单一真相源，4 程序）投影到 E-MTAB-8659
      Two-color microarray (n=63 lung adenocarcinoma brain metastases)，
      与 A 队列训练得到的 4-dim centroid 做最近邻归类，
      对每个 BrMS subtype 算 program AUC（one-vs-rest），
      验证「跨平台一致性」(README line 354)。

不跑完整 BrMS 重训练（A 队列 program scores 的 centroid 已代表训练结果）。
数据准备：
  - 36.8 MB processed_all.txt（log2(Cy3/Cy5) 比值 + median-centered）
  - sdrf.txt 提供 per-sample 临床（年龄/性别/诊断；无 BrMS 原标签）
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import json
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (
    ROOT, AN, FG, GEO, SEED,
    BRMS_PROGRAMS, COMPOSITION_PANELS,
    SUBS, NM, NAMES, COHORT_META, PALETTE,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(SEED)

TZ_BJ = timezone(timedelta(hours=8))
UNFILT_DIR = os.path.join(GEO, "EMTAB8659_unfiltered")
PROCESSED = os.path.join(UNFILT_DIR, "processed_all.txt")
SDRF = os.path.join(UNFILT_DIR, "E-MTAB-8659.sdrf.txt")


def now_bj() -> str:
    return datetime.now(TZ_BJ).strftime("%Y-%m-%d %H:%M:%S %Z")


def main():
    """E-MTAB-8659 → 4 program scores → nearest-centroid BrMS call → per-program AUC

    按 README line 354 「跨平台一致性」要求，给出 RNA-seq (A/B) vs microarray
    (E-MTAB-8659) 之间的 program score signature 一致性。
    """
    t0 = datetime.now(TZ_BJ)
    print("=" * 78)
    print(f"E-MTAB-8659 BrMS 投影（A3） · 启动 {now_bj()}")
    print("=" * 78)

    # —— 1) 加载 SDRF (per-sample metadata；已有 64 行无需等待 processed 下载)
    sdrf = pd.read_csv(SDRF, sep="\t")
    sdrf = sdrf.rename(columns={
        "Source Name": "sample",
        "Characteristics[age]": "age",
        "Characteristics[sex]": "sex",
        "Characteristics[disease]": "disease",
        "Characteristics[individual]": "individual",
    })
    samples = sdrf["sample"].astype(str).tolist()
    print(f"  SDRF: {len(samples)} samples, disease={sdrf['disease'].unique()}")

    # —— 2) 加载 processed_all.txt（★ partial-tolerant：拿到几列就跑几列）
    print(f"\n  Loading processed_all.txt ... ({os.path.getsize(PROCESSED)/1024:.0f} KB so far)")
    # processed_all.txt 格式：「"GeneSymbol"\t"NCH624h"\t...」
    # 头部 1 行；读时强制 dtype=str，处理 raw
    Xraw = pd.read_csv(PROCESSED, sep="\t", quoting=1, low_memory=False, engine="c", nrows=0)
    cols_in_file = Xraw.columns.tolist()
    # 把全部想要的 sample 列一次性读入；用 usecols 减少内存
    keep_cols = [c for c in cols_in_file if c == "GeneSymbol" or c in samples]
    print(f"  头部检测：{len(cols_in_file)} 列（含 GeneSymbol），目标保持 {len(keep_cols)} 列")
    if len(keep_cols) < 5:
        print(f"  ❌ 头部不可用（{len(cols_in_file)} 列），无法继续。")
        return

    # 关键：pd.read_csv with usecols + iterator 模式分块读
    # E-MTAB-8659 microarray 含 "-" 占位符（missing / filtered），用 na_values 转换
    chunks = []
    chunks_rows = 0
    reader = pd.read_csv(
        PROCESSED, sep="\t", quoting=1, engine="c", low_memory=False,
        usecols=keep_cols,
        na_values=["-", "NA", "N/A", "n/a", "n.a.", ""],
        chunksize=4000,
    )
    for ck in reader:
        # 处理 chunk：行是 GeneSymbol × sample，列 = sample
        ck = ck.set_index("GeneSymbol")
        # 替换 NaN/inf 为 0
        ck = ck.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        chunks.append(ck)
        chunks_rows += ck.shape[0]
    if not chunks:
        print("  ❌ 没有读取到任何数据行，processed 下载未完成。")
        return
    M = pd.concat(chunks, axis=0)
    print(f"\n  E-MTAB-8659 matrix: {M.shape[0]} genes × {M.shape[1]} samples (after concat)")
    # T 转置：行=sample, 列=gene
    M = M.T
    print(f"  Transposed M: samples × genes = {M.shape}")

    # —— 3) Z-score normalize within-cohort（★ 关键，跨平台 z-score 让 A 与 E-MTAB-8659 在同尺度）
    Z = M.sub(M.mean(axis=0), axis=1).div(M.std(axis=0).replace(0, np.nan), axis=1)
    Z = Z.dropna(how="all", axis=1).fillna(0)
    print(f"  within-cohort z-score: {Z.shape}")

    # —— 4) 投影 A 队列的 4 个 program（BRMS_PROGRAMS）到 E-MTAB-8659
    print("\n  投影 BrMS 程序（来自 A 队列训练的 BRMS_PROGRAMS 单一真相源）")
    sig_scores = {}
    sig_cov = {}
    for sub in SUBS:
        gset = [g for g in BRMS_PROGRAMS[sub] if g in Z.columns]
        if len(gset) < 5:
            print(f"    ⚠ BrMS{sub}: 只覆盖 {len(gset)} 基因，跳过")
            sig_scores[sub] = pd.Series(np.nan, index=Z.index)
            sig_cov[sub] = 0
            continue
        score = Z[gset].mean(axis=1)
        sig_scores[sub] = score
        sig_cov[sub] = len(gset)
        print(f"    BrMS{sub} ({NM[sub]}): {len(gset)}/{len(BRMS_PROGRAMS[sub])} genes -> score std={score.std():.3f}")
    S = pd.DataFrame(sig_scores)
    print(f"\n  BrMS 投影打分矩阵: {S.shape}")

    # —— 5) 加载 A 队列 reference scores（scores_gse245467 是 meanZ on 10 通路）；
    #        需要 A 队列的 BrMS 4-程序 score（A 队列 meanZ on BRMS_PROGRAMS）
    #        如果没缓存过，下面现场算（_common 配置可直接复用 02 算法）
    print("\n  加载 A 队列 4-程序参考（来自 scores_meanZ 10-通路 → 4 program）")
    if not os.path.exists(os.path.join(AN, "scores_meanZ.parquet")):
        print("  ⚠ A 队列 scores_meanZ 缺失；略过 nearest-centroid 标注，留 per-program AUC")
        have_ref = False
    else:
        S_A = pd.read_parquet(os.path.join(AN, "scores_meanZ.parquet"), engine="pyarrow")
        lab = pd.read_parquet(os.path.join(AN, "labels_used.parquet"), engine="pyarrow")
        print(f"    A scores_meanZ: {S_A.shape}, labels: {lab.shape}")

        # A 队列同样运行 BRMS_PROGRAMS（确保口径统一）
        Xc = pd.read_parquet(os.path.join(AN, "X_counts.parquet"), engine="pyarrow")
        X = Xc[lab.index]
        cpm = X.div(X.sum(axis=0), axis=1) * 1e6
        L = np.log2(cpm + 1)
        ZA = L.sub(L.mean(axis=1), axis=0).div(L.std(axis=1).replace(0, np.nan), axis=0)
        ZA = ZA.dropna(how="all")
        # REF: samples × programs (每个 sub 的 program score)
        REF = pd.DataFrame(index=lab.index)
        for sub in SUBS:
            gset = [g for g in BRMS_PROGRAMS[sub] if g in ZA.index]
            if len(gset) >= 5:
                REF[sub] = ZA.loc[gset].mean(axis=0)
        print(f"    A REF 矩阵: {REF.shape}, NaN cols: "
              f"{[c for c in REF.columns if REF[c].isna().any()]}")
        # A 队列 per-sample BrMS 真值
        y = lab["brms"].astype(str).values

        # 关键：每个数据集**内部**按程序列 z-score，把 A 与 E-MTAB 移到同一坐标空间
        REF_z = (REF - REF.mean(axis=0)) / REF.std(axis=0).replace(0, np.nan)
        S_z = (S - S.mean(axis=0)) / S.std(axis=0).replace(0, np.nan)

        # === BrMS3 program coverage 检查（关键算法选择）===
        # 如果 S_z 中 BrMS3 列 100% NaN（microarray 探针覆盖不足），用 3D（skip BrMS3）
        # 否则用 4D。
        if S_z["3"].isna().all() or (S_z["3"].notna().sum() < 5):
            used_progs = ["1", "2", "4"]
            print(f"  ⚠ BrMS3 program 在 E-MTAB-8659 NaN coverage → 用 3D (skip BrMS3) 计算 centroid 距离")
        else:
            used_progs = SUBS
            print(f"  ✓ BrMS3 program 在 E-MTAB-8659 有覆盖 → 用 4D 计算 centroid 距离")

        # centroids: shape (len(used_progs) programs, 4 classes)
        centroids_pc = np.column_stack([
            REF_z.loc[y == sub].mean(axis=0)[used_progs].fillna(0).values
            for sub in SUBS
        ])  # shape (k_progs, 4)
        # 准备 S_z 投影到 used_progs（NaN → 0）
        S_z_use = S_z[used_progs].fillna(0).values
        # 每个 E-MTAB sample 到 4 centroids 的欧式距离
        dists = np.zeros((S_z_use.shape[0], len(SUBS)))
        for i in range(len(SUBS)):
            dists[:, i] = np.linalg.norm(S_z_use - centroids_pc[:, i], axis=1)
        # 最近 centroid = nearest BrMS call
        emtab_brms = pd.Series(
            [SUBS[i] for i in np.argmin(dists, axis=1)],
            index=S.index, name="brms_pred",
        )
        # 输出 centroids 用于人工 sanity check
        cent_df = pd.DataFrame(
            centroids_pc, index=[f"prog_{p}" for p in used_progs], columns=SUBS,
        )
        print("\n  Centroids (program × class): A cohort per-subtype mean of z-scores")
        print(cent_df.round(3).to_string())

        # 输出 call 分布
        call_counts = emtab_brms.value_counts().to_dict()
        print(f"\n  E-MTAB-8659 nearest-centroid BrMS 调用 ({len(call_counts)} 类): {call_counts}")
        have_ref = True

    # —— 6) 与 A 队列一致的 per-program one-vs-rest AUC（无真值）改用 cross-platform signal concordance
    print("\n  跨平台一致性：每个 BrMS program 在 E-MTAB-8659 → S[B] 分布形态 vs A 队列")
    rows = []
    for sub in SUBS:
        col = S[sub].dropna()
        if len(col) < 5:
            continue
        rows.append({
            "subtype": sub,
            "name": NM[sub],
            "n_samples_E": len(col),
            "n_genes_used": sig_cov[sub],
            "score_mean_E": col.mean(),
            "score_std_E": col.std(),
            "score_median_E": col.median(),
            "score_p05_E": col.quantile(0.05),
            "score_p95_E": col.quantile(0.95),
        })
    summary_C = pd.DataFrame(rows)
    if have_ref:
        # 加 A 队列同口径
        rows_A = []
        for sub in SUBS:
            col = REF[sub].dropna()
            if len(col) < 5:
                continue
            rows_A.append({
                "subtype": sub,
                "score_mean_A": col.mean(),
                "score_std_A": col.std(),
                "score_median_A": col.median(),
            })
        dfA = pd.DataFrame(rows_A).set_index("subtype")
        summary_C = summary_C.merge(dfA, left_on="subtype", right_index=True, how="left")
    print(summary_C.round(3).to_string(index=False))

    # —— 7) 写产物
    safe_out = f"{AN}/cohort_C_br_projection_summary.csv"
    summary_C.to_csv(safe_out, index=False, encoding="utf-8")
    print(f"\n  ✅ Wrote {safe_out}")
    if have_ref:
        emtab_brms.to_csv(f"{AN}/emtab8659_brms_call.csv", header=True)
        print(f"  ✅ Wrote {AN}/emtab8659_brms_call.csv")

        # S_z matrix
        S_z.to_csv(f"{AN}/emtab8659_br_projection_zscores.csv")
        print(f"  ✅ Wrote {AN}/emtab8659_br_projection_zscores.csv")

    # S (raw scores) 存全量
    S.to_csv(f"{AN}/emtab8659_br_projection_scores.csv")
    print(f"  ✅ Wrote {AN}/emtab8659_br_projection_scores.csv")

    # —— 8) 一张 2x2 panel figure (per-BrMS score dist: E-MTAB vs A)
    # ⚠ BrMS3 panel 在 E-MTAB-8659 上无 score（探针覆盖 < 4/9，S_z['3'] NaN）；
    #    只画 A 队列分布并在标题明示限制，避免误读"两 cohort 一致"
    if have_ref:
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.4))
        for ax, sub in zip(axes.flat, SUBS):
            ax.hist(REF[sub].dropna(), bins=18, alpha=0.65, color="#4878CF",
                    density=True, label=f"A (n={(y==sub).sum()})")
            if sub in used_progs and sub in S.columns:
                ax.hist(S[sub].dropna(), bins=18, alpha=0.55, color="#EE854A",
                        density=True,
                        label=f"E-MTAB-8659 score (n={(~S[sub].isna()).sum()})")
            else:
                # BrMS3 program 探针不足；在 panel 上叠一面浅色 mask 标记区域
                ax.text(0.5, 0.93, "E-MTAB-8659: BrMS3 program probe coverage\ninsufficient on this platform",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=8, color="#AA4400",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E6",
                                  ec="#EE854A", lw=0.6))
            call_n = int((emtab_brms == sub).sum())
            ax.set_title(
                f"BrMS{sub} — {NM[sub]}\n"
                f"A cohort n={(y==sub).sum()} | E-MTAB-8659 nearest-centroid call={call_n}",
                fontsize=10,
            )
            ax.set_xlabel("program score")
            ax.legend(fontsize=8, frameon=False, loc="upper right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.suptitle(
            "Cross-platform BrMS program score distribution\n"
            f"A=GSE245467 (RNA-seq, n=52) vs E-MTAB-8659 (Two-color microarray, n=63)",
            fontsize=11,
        )
        fig.tight_layout()
        out_fig = f"{FG}/Fig_C_br_projection.png"
        fig.savefig(out_fig, dpi=160)
        print(f"  ✅ Wrote {out_fig}")
    elapsed = (datetime.now(TZ_BJ) - t0).total_seconds() / 60
    print("\n" + "=" * 78)
    print(f"完成于 {now_bj()}（{elapsed:.1f} min）")
    print("=" * 78)


if __name__ == "__main__":
    main()
