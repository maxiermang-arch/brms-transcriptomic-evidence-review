"""analysis/_common.py — 神经外科 BrMS 审计项目共享配置

本模块由 01–19 全部脚本统一引用，避免：
  • 路径硬编码（P0-2）
  • 跨脚本重复定义 SUBS/NM/BrMS2 程序/成分面板 等常量（P1-4、P0-6）
  • 共享路径、PALETTE 等"真相散落"问题

约定：本文件**只放配置与无副作用的常量/工具**，不执行任何 import 时的副作用。
"""
from __future__ import annotations
import json
import os
import pickle
from typing import Any

# ============ 路径 ============
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ANALYSIS_DIR)
DATA = os.path.join(ROOT, "data")
AN = ANALYSIS_DIR
PU = os.path.join(DATA, "purity")
GEO = os.path.join(DATA, "geo")
SUPP = os.path.join(DATA, "supplementary", "41467_2026_68748_MOESM3_ESM.xlsx")
GENESETS = os.path.join(AN, "genesets")
FG = os.path.join(ROOT, "figures")

# ============ 亚型与队列 ============
SUBS = ["1", "2", "3", "4"]
NM = {
    "1": "BrMS1 neural",
    "2": "BrMS2 immune/EMT",
    "3": "BrMS3 metabolic",
    "4": "BrMS4 prolif.",
}
NAMES = {"1": "BrMS1", "2": "BrMS2", "3": "BrMS3", "4": "BrMS4"}
SUB_LABELS_PRETTY = ["BrMS1\nneural", "BrMS2\nimmune/EMT",
                     "BrMS3\nmetabolic", "BrMS4\nprolif."]
COHORT_META = {
    "A": ("GSE245467", "mixed cancer origins", 52),
    "B": ("GSE184869", "breast", 45),
}

# ============ 随机种子 ============
SEED = 20260913

import numpy as _np
RNG = _np.random.default_rng(SEED)  # 旧脚本兼容：np.random.default_rng 移到此处

# ============ 配色（统一调色板）============
PALETTE = {
    "A": "#4878CF",       # cohort A — melanoma
    "B": "#EE854A",       # cohort B — breast
    "subtypes": ["#4C72B0", "#DD8452", "#55A868", "#C44E52"],  # BrMS1–4
}


# ============ 组织成分面板（9 组共 88 基因；单一真相源，P0-6）============
# 与"程序基因集"概念不同——这些是组织/细胞类型标记，用于纯度校正。
COMPOSITION_PANELS = {
    "脑组织-星形胶质": ["GFAP", "AQP4", "SLC1A2", "SLC1A3", "ALDH1L1",
                         "GJA1", "S100B", "SOX9", "NFIA", "SLC4A4"],
    "脑组织-神经元":   ["RBFOX3", "SYP", "SNAP25", "NEFL", "NEFM", "NEFH",
                         "MAP2", "SCN2A", "RELN", "GABRA1", "GRIN1", "SYN1",
                         "CAMK2A", "SYT1", "GRIA2", "KCNJ4", "NECAB1", "ERMN"],
    "脑组织-少突胶质": ["PLP1", "MBP", "MOBP", "MAG", "CNP", "MOG",
                         "CLDN11", "OPALIN", "KLK6"],
    "脑组织-小胶质":   ["P2RY12", "TMEM119", "CX3CR1", "SALL1", "GPR34"],
    "免疫-泛白系":     ["PTPRC", "LCP2", "HCLS1", "ARHGDIB", "CORO1A",
                         "CD48", "CD52", "CD53"],
    "免疫-T/NK":      ["CD3D", "CD3E", "CD2", "CD8A", "CD8B", "NKG7",
                         "GNLY", "KLRD1", "IL2RB", "TRAC"],
    "免疫-髓系":       ["CD68", "CD14", "AIF1", "LYZ", "TYROBP",
                         "FCER1G", "C1QA", "C1QB", "ITGAM", "CSF1R"],
    "基质-成纤维":     ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM",
                         "FAP", "PDGFRB", "ACTA2", "FBN1", "POSTN", "THY1"],
    "上皮-肿瘤":       ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1",
                         "CLDN3", "CLDN4", "MUC1", "KRT7", "ELF3"],
}


def _hallmark():
    """懒加载 hallmark.json（避免模块级副作用）"""
    return json.load(open(os.path.join(GENESETS, "hallmark.json")))


def _neural():
    return json.load(open(os.path.join(GENESETS, "neural.json")))


# ============ BrMS 程序基因集（★ 单一真相源，修正 P0-6）============
# BrMS1/4 程序来自 hallmark/neural.json；BrMS2/3 此前在 14 与 16 各写一份，
# 现统一固化在此（修改请只改这里）。
BRMS_PROGRAMS = {
    # BrMS1 neural — 用神经投影导向通路（与 09_survival 的 PANELS 同源）
    "1": list(_neural()["Neuron Projection Guidance (GO:0097485)"]),
    # BrMS2 immune — 此前在 14:129 和 16:47 各写一份，现合并
    "2": ["CD3D", "CD3E", "CD3G", "CD2", "CD247", "TRAC", "LCK", "ZAP70",
          "CD8A", "CD8B", "CD68", "CD14", "AIF1", "LYZ", "TYROBP",
          "FCER1G", "C1QA", "C1QB"],
    # BrMS3 epithelial
    "3": ["CDH1", "EPCAM", "KRT8", "KRT18", "KRT19", "CLDN3", "CLDN4",
          "OCLN", "TJP1"],
    # BrMS4 prolif. — Hallmark E2F + G2-M 并集
    "4": list(set(_hallmark()["E2F Targets"]) | set(_hallmark()["G2-M Checkpoint"])),
}


# ============ Fig6 A panel "defining program" 映射（P0-4 唯一真相源）============
# Fig6 A panel 硬编码数字 c1/c2 的语义：每个亚型取一个"代表程序"算 one-vs-rest AUC。
# 映射规则：
#   BrMS1 neural  → Neuron Projection Guidance (GO:0097485)  （神经突触导向）
#   BrMS2 immune  → Allograft Rejection（免疫学特征最强）
#   BrMS3 metab.  → Fatty Acid Metabolism（代谢定义核心通路）
#   BrMS4 prolif. → E2F Targets（细胞周期/增殖最强单程序，比 E2F+G2M 并集更紧扣 Fig6 原硬编码）
# 改这个映射请同时更新 Fig6 / README / 论文 §Methods。
FIG6_REP_PROGRAM = {
    "1": ("neural.json", "Neuron Projection Guidance (GO:0097485)"),
    "2": ("hallmark.json", "Allograft Rejection"),
    "3": ("hallmark.json", "Fatty Acid Metabolism"),
    "4": ("hallmark.json", "E2F Targets"),
}


# ============ ABIS 行顺序 ============
ABIS_CELLTYPES = [
    "Monocytes C", "NK", "T CD8 Memory", "T CD4 Naive", "T CD8 Naive",
    "B Naive", "T CD4 Memory", "MAIT", "T gd Vd2", "Neutrophils LD",
    "T gd non-Vd2", "Basophils LD", "Monocytes NC+I", "B Memory",
    "mDCs", "pDCs", "Plasmablasts",
]


# ============ 工具函数 ============

def load_pickle(path: str, expected_keys: list[str] | None = None) -> Any:
    """带 schema 校验的 pickle 装载（P0-3 部分缓解：默认不信任）。

    expected_keys: 若提供，校验顶层 dict 包含这些键；防止跨脚本字段漂移。

    ⚠️ P0-3 状态（2026-09-13）：
        pickle 反序列化有 RCE 风险，**只信任本机自己产出的文件**。
        当前分析链路仍以 pickle 为跨脚本数据传递格式；P1 计划：
          - 将跨脚本产物全部迁移到 .parquet（pandas.to_parquet/read_parquet）
          - 本函数保留为"过渡期 wrapper"，仍按当前行为工作
        临时：每次装载打印一行 DeprecationWarning 提醒。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"pickle 不存在：{path}（上游脚本是否跑过？）")
    import warnings as _w
    _w.warn(
        f"load_pickle 仍在使用 .pkl（{os.path.basename(path)}）；"
        f"P1 将迁移到 .parquet。",
        category=DeprecationWarning,
        stacklevel=2,
    )
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if expected_keys is not None and isinstance(obj, dict):
        missing = [k for k in expected_keys if k not in obj]
        if missing:
            raise KeyError(
                f"{path}: 缺少字段 {missing}（期望 {expected_keys}，实际 {list(obj)}）")
    return obj


# ============ 数据装载（P1-1 + P2：parquet 强制，无 .pkl 兜底）============
# P2 状态（2026-09-13）：
#   - 9/9 .pkl 已迁 .parquet（X_counts / auc_matrix / labels_{all,gse245467,used} /
#     scores_{aucell,gse245467,meanZ} + gse184869_symbol_matrix 三件套）
#   - load_dataframe / load_pkl_tuple **不再** 兜底读 .pkl——遇到只剩 .pkl 的情形
#     会直接 raise FileNotFoundError，引导用户跑 00_pickle_to_parquet.py
#   - 紧急逃生：设环境变量 NEUROSURG_ALLOW_PKL=1 可临时启用兜底（仅供单次回滚）
#
# 历史：P1-1（2026-09-13）首版；P1-Fix-14（2026-09-13 14 行 .values 维度错误已修）


def load_dataframe(path: str) -> pd.DataFrame:
    """parquet 强制装载（**P2** 已移除 .pkl 兜底）。

    参数 `path` 习惯上仍写 `.pkl` 后缀（不改消费端路径文本），函数内部
    自动替换为 `.parquet`。若 `.parquet` 不存在则直接 raise FileNotFoundError。

    Raises:
        FileNotFoundError: `.parquet` 缺失且未设 NEUROSURG_ALLOW_PKL=1。
        RuntimeError: 旧 .pkl 文件仍残留，引导迁移。
    """
    import pandas as _pd
    base, _ = os.path.splitext(path)
    pq = base + ".parquet"
    if os.path.exists(pq):
        return _pd.read_parquet(pq, engine="pyarrow")
    if os.path.exists(path):
        # .pkl 仍残留：P2 起 fail-fast
        allow = os.environ.get("NEUROSURG_ALLOW_PKL") == "1"
        if not allow:
            raise RuntimeError(
                f"load_dataframe 检测到遗留 .pkl 但无 .parquet：{path}\n"
                f"  → 请跑 `python analysis/00_pickle_to_parquet.py`（默认 DRY 检查；\n"
                f"    再加 `--execute` 真正迁移；建议先 `--backup` 备份）\n"
                f"  → 紧急回滚：设 `NEUROSURG_ALLOW_PKL=1` 临时启用 .pkl 兜底（不推荐）"
            )
        # 紧急回滚路径：单次 .pkl 装载
        return _pd.read_pickle(path)
    raise FileNotFoundError(
        f"load_dataframe: .parquet 与 .pkl 都不存在（{base}）")


def load_pkl_tuple(path: str) -> tuple:
    """tuple pickle → 解包为三件套 .parquet（X / labels / os）（**P2** 无 .pkl 兜底）。

    用于 `gse184869_symbol_matrix.pkl` 这种 (X, lab, os) 元组。
    配套产物：{base}.X.parquet + {base}.labels.parquet + {base}.os.parquet
              + {base}.manifest.json（由 00_pickle_to_parquet.py 生成）。
    """
    import pandas as _pd
    base, _ = os.path.splitext(path)
    pq_X = base + ".X.parquet"
    pq_lab = base + ".labels.parquet"
    pq_os = base + ".os.parquet"
    manifest = base + ".manifest.json"
    if all(os.path.exists(p) for p in (pq_X, pq_lab, pq_os, manifest)):
        return (
            _pd.read_parquet(pq_X, engine="pyarrow"),
            _pd.read_parquet(pq_lab, engine="pyarrow"),
            _pd.read_parquet(pq_os, engine="pyarrow"),
        )
    if os.path.exists(path):
        # .pkl 仍残留：P2 起 fail-fast
        allow = os.environ.get("NEUROSURG_ALLOW_PKL") == "1"
        if not allow:
            raise RuntimeError(
                f"load_pkl_tuple 检测到遗留 .pkl 但三件套 .parquet 缺失：{path}\n"
                f"  → 请跑 `python analysis/00_pickle_to_parquet.py`（默认 DRY 检查；\n"
                f"    再加 `--execute` 真正迁移；建议先 `--backup` 备份）\n"
                f"  → 紧急回滚：设 `NEUROSURG_ALLOW_PKL=1` 临时启用 .pkl 兜底（不推荐）"
            )
        # 紧急回滚路径：单次 pickle 装载
        with open(path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(
        f"load_pkl_tuple: 三件套 .parquet 与 .pkl 都不存在（{base}）")


def safe_makedirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def setup_paths() -> tuple[str, str, str, str, str, str]:
    """返回 (ROOT, AN, PU, FG, DATA, SUPP)。"""
    return ROOT, AN, PU, FG, DATA, SUPP
