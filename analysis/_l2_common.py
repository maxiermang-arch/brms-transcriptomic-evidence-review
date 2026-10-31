# -*- coding: utf-8 -*-
"""analysis/_l2_common.py — L2（组织成分校正）层共享机制

建立原因（2026-09-15，响应投稿前预审）：
  《16_final_purity_verdict.py》与《41_L2_sensitivity.py》各自重复实现了一遍
  L2 的装载 / 打分 / 校正流程，且两处实现并不一致：

    * 16 用 `C['brms'].astype(str)` 解析标签 —— 队列 A 的 brms 列是 float，
      得到 `'1.0'`，与 SUBS = ['1'…'4'] 恒不匹配 → 队列 A 的 `y` 全为 0，
      `roc_auc_score` 静默失败 → CSV 里队列 A 四行的 AUC 全部空白，而 R²
      仍然写出（OLS 不依赖 y）。稿件据此把空白误读成"队列 A 纯度估计不完整"。
    * 16 的"留出自身域"只排除**以自身域命名的成分面板**，五个 ESTIMATE 派生列
      从不排除 → 对 BrMS2（免疫域）而言模型中仍保留独立的免疫含量测量，
      leave-own-domain-out 并未真正实现。

本模块把这两处的正确实现集中成单一真相源，供 16 / 41 共同引用，
避免再出现"两处各写一份、结果对不上"。

只放常量与无副作用的函数；不在 import 时读写任何数据文件。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import AN, PU, SUBS, BRMS_PROGRAMS, SEED, load_dataframe, load_pkl_tuple  # noqa: E402


# ---------------------------------------------------------------- 协变量定义
# 成分面板（四个"合计"列 + 基质/上皮为单列面板），与 ESTIMATE 五列
PANELS = ["脑组织合计", "免疫合计", "基质-成纤维", "上皮-肿瘤"]
ESTIMATE = ["ESTIMATE_Stromal", "ESTIMATE_Immune", "ESTIMATE_score", "Purity_ESTIMATE"]

# 每个亚型"同名"的成分面板（即必须留出的那一列）
OWN_PANEL = {"1": ["脑组织合计"], "2": ["免疫合计"], "3": ["上皮-肿瘤"], "4": []}

# 与该面板测量同一域的 ESTIMATE 派生列（严格版一并留出）
#   1 神经：ESTIMATE 没有神经轴 → 无
#   2 免疫：ESTIMATE_Immune 直接测免疫；ESTIMATE_score / Purity_ESTIMATE 由
#           Stromal+Immune 合成 → 含免疫信息，一并留出
#   3 上皮：ESTIMATE_Stromal（基质/间质，与上皮-肿瘤区室互补）与 Purity_ESTIMATE
#   4 增殖：不是组织成分域 → 无
OWN_ESTIMATE = {
    "1": [],
    "2": ["ESTIMATE_Immune", "ESTIMATE_score", "Purity_ESTIMATE"],
    "3": ["ESTIMATE_Stromal", "Purity_ESTIMATE"],
    "4": [],
}

# 五种调整规格（顺序即报告顺序）
SPECS = ["full", "leaveown_current", "leaveown_strict", "purity_only", "no_estimate"]


def adjustment_sets(s: str, columns=None) -> dict[str, list[str]]:
    """返回某亚型在五种规格下的协变量清单（未做存在性/缺失值过滤）。"""
    own_p, own_e = OWN_PANEL[s], OWN_ESTIMATE[s]
    return {
        "full": PANELS + ESTIMATE,
        "leaveown_current": [c for c in PANELS if c not in own_p] + ESTIMATE,
        "leaveown_strict": ([c for c in PANELS if c not in own_p]
                            + [c for c in ESTIMATE if c not in own_e]),
        "purity_only": ["Purity_ESTIMATE"],
        "no_estimate": [c for c in PANELS if c not in own_p],
    }


def usable(cols: list[str], C: pd.DataFrame) -> list[str]:
    """过滤到实际存在且无缺失的协变量（ESTIMATE 在部分样本上会缺列/缺值）。"""
    return [c for c in cols if c in C.columns and C[c].notna().all()]


# ---------------------------------------------------------------- 统计量
def _design(covars: np.ndarray) -> np.ndarray:
    """把协变量 z 标准化后加截距列。

    ESTIMATE 三列量纲悬殊（Stromal / Immune 为 0–数千，Purity 为 0–1），
    直接用原始尺度会让设计矩阵条件数爆表、statsmodels 报秩亏告警。
    由于含截距的 OLS 残差对协变量的**可逆仿射变换不变**，标准化不改变
    任何残差 AUC，只改善数值条件。标准差为 0 的列退化为常数（跳过缩放）。
    """
    X = np.asarray(covars, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return np.column_stack([np.ones(len(X)), (X - mu) / sd])


def _resid(prog: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, float]:
    """OLS 残差 + 决定系数 + 设计矩阵条件数（用于披露共线性）。"""
    yv = np.asarray(prog, float).ravel()
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    res = yv - X @ beta
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - float((res ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    cond = float(np.linalg.cond(X))
    return res, r2, cond


def residual_auc(prog: np.ndarray, covars: np.ndarray, y: np.ndarray) -> float:
    """样本内残差 AUC（OLS 拟合后取残差对标签的 AUC）。"""
    res, _, _ = _resid(prog, _design(covars))
    return float(roc_auc_score(y, res))


def rank_info(covars: np.ndarray) -> tuple[int, int]:
    """返回 (数值秩, 含截距的设计矩阵列数)。

    `full` 规格必然秩亏 1 列：ESTIMATE_score ≡ ESTIMATE_Stromal + ESTIMATE_Immune
    （实测 R² = 1.000000，系数恰为 1, 1），这是 ESTIMATE 的定义而非缺陷。
    秩亏时最小范数解给出的残差仍是 y 在 col(X) 上**唯一**的正交投影残差，
    所以残差 AUC / R² 依旧有效，只是单个系数不可识别（本文不报告系数）。
    """
    X = _design(covars)
    return int(np.linalg.matrix_rank(X)), int(X.shape[1])


def fit_diag(prog: np.ndarray, covars: np.ndarray, y: np.ndarray) -> dict:
    """一次拟合返回残差 AUC / R² / 条件数（供 16 号脚本落表）。"""
    res, r2, cond = _resid(prog, _design(covars))
    return {"auc": float(roc_auc_score(y, res)), "r2": r2, "cond": cond}


def crossfit_auc(prog: np.ndarray, covars: np.ndarray, y: np.ndarray,
                 n_splits: int = 5) -> float:
    """交叉拟合残差 AUC：OLS 只在训练折拟合，系数套用到留出折。

    将残差拟合与评估分离；偏差不保证为正，仍需校准和不确定性评估。
    """
    X = np.asarray(covars, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    res = np.empty(len(y))
    for tr, te in skf.split(X, y):
        Xtr = _design(X[tr])
        beta, *_ = np.linalg.lstsq(Xtr, np.asarray(prog, float).ravel()[tr], rcond=None)
        # 用训练折的均值/标准差标准化留出折，再用训练折系数外推
        Xte = np.column_stack([np.ones(len(te)), (X[te] - X[tr].mean(axis=0)) / _sd(X[tr])])
        res[te] = np.asarray(prog, float).ravel()[te] - Xte @ beta
    return float(roc_auc_score(y, res))


def _sd(X: np.ndarray) -> np.ndarray:
    sd = X.std(axis=0)
    return np.where(sd > 0, sd, 1.0)


def boot_ci(prog: np.ndarray, covars: np.ndarray, y: np.ndarray,
            n: int = 2000, seed: int = SEED) -> tuple[float, float]:
    """嵌套 bootstrap：每次重采样内部重新拟合 OLS 再算残差 AUC。"""
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    idx_all = np.arange(len(y))
    for i in range(n):
        idx = rng.choice(idx_all, size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            vals[i] = np.nan
            continue
        vals[i] = residual_auc(prog[idx], covars[idx], y[idx])
    vals = vals[~np.isnan(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- 数据装载
def _z(L: pd.DataFrame) -> pd.DataFrame:
    return (L.sub(L.mean(axis=1), axis=0)
             .div(L.std(axis=1).replace(0, np.nan), axis=0).dropna(how="all"))


def load_cohorts() -> dict[str, dict]:
    """装载两队列的成分表、标签与 z 分数表达矩阵，并保证三者样本对齐。

    返回 {cohort: {'C': 成分表(n×p), 'lab': 标签 Series(str),
                   'Z': z 分数(基因×样本), 'dropped': 被剔除的样本}}。

    两处历史坑（均在此统一处理，勿在下游脚本里重复踩）：
      1. brms 列在队列 A 是 float → `astype(str)` 得 `'1.0'`，恒不匹配 SUBS。
         这里归一为裸整数字符串。
      2. 队列 A 的 BREBRM21-B / BREBRM46-B 有有效计数，但缺少可用 BrMS 标签。
         上述标签筛选先排除它们；对齐完整成分表后相应列为 NaN，并非原始计数为空。
    """
    out: dict[str, dict] = {}
    for co in ("A", "B"):
        C = pd.read_csv(f"{PU}/cohort{co}_composition.txt", sep="\t", index_col=0)
        lab = C["brms"].map(lambda v: "nan" if pd.isna(v) else str(int(v)))
        C = C.drop(columns=["brms"])
        out[co] = {"C": C, "lab": lab}

    XA = load_dataframe(f"{AN}/X_counts.pkl")
    lA = load_dataframe(f"{AN}/labels_gse245467.pkl")
    lA = lA.iloc[:, 0] if isinstance(lA, pd.DataFrame) else lA
    kA = lA[lA.isin(SUBS)].index
    LA = np.log2(XA[kA] / XA[kA].sum(axis=0) * 1e6 + 1)

    XB, lB, osB = load_pkl_tuple(f"{AN}/gse184869_symbol_matrix.pkl")
    lB = lB.iloc[:, 0] if isinstance(lB, pd.DataFrame) else lB
    kB = lB[lB.isin(SUBS)].index
    # 队列 B 矩阵已是 log2 尺度，不再取 log
    LB = XB[kB]

    LZ = {"A": _z(LA), "B": _z(LB)}
    OS = {"B": osB}

    for co in ("A", "B"):
        C, Zc = out[co]["C"], LZ[co].reindex(columns=out[co]["C"].index)
        keep = [s for s in C.index if s in Zc.columns and not Zc[s].isna().all()]
        dropped = [s for s in C.index if s not in keep]
        out[co]["C"] = C.loc[keep]
        out[co]["lab"] = out[co]["lab"].loc[keep]
        out[co]["Z"] = Zc[keep]
        out[co]["dropped"] = dropped
    out["B"]["os"] = OS["B"]
    return out


def program_genes(Z: pd.DataFrame, s: str) -> list[str]:
    """某亚型程序在该队列里实际可用的基因（处理平台差异与全 NaN 行）。"""
    return [g for g in BRMS_PROGRAMS[s]
            if g in Z.index and not Z.loc[g].isna().all()]


def program_score(Z: pd.DataFrame, s: str) -> np.ndarray:
    g = program_genes(Z, s)
    return Z.loc[g].mean(axis=0).values
