#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""00 · 一次性把现存 .pkl 产物迁移到 .parquet（P1-1）

设计：
  - 默认 dry-run，只打印 + 写 migration_report.json（带 src → dst 映射与校验）
  - `--execute` 才真正写 .parquet
  - DataFrame（pandas）→ 单文件 .parquet
  - tuple / list（gse184869_symbol_matrix.pkl 是 (X, lab, os)）→ 解包
       + 同 basename_stem.parquet + .labels.parquet + .os.parquet + manifest.json
  - 校验：转完后读回对比 shape/dtype 与 NaN 数

依赖：pandas>=2.2 + pyarrow>=15（已装在 venv default）

用法：
      python 00_pickle_to_parquet.py            # dry-run
      python 00_pickle_to_parquet.py --execute  # 真做
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd

# —— 共享配置（P0-2 去硬编码路径）——
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import AN  # noqa: E402

REPORT_PATH = os.path.join(AN, "migration_report.json")
BACKUP_DIR = "/tmp/p1_backup_pkls"


def _md5(path: str) -> str:
    """文件 MD5（仅取前 16 hex）。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _shape_eq(a: Any, b: Any) -> bool:
    try:
        return a.shape == b.shape
    except Exception:
        return a == b


def migrate_pandas_object(src: str, execute: bool) -> dict:
    """DataFrame / Series pickle → 单 .parquet。"""
    dst = src[:-4] + ".parquet"
    obj = pickle.load(open(src, "rb"))
    if not isinstance(obj, (pd.DataFrame, pd.Series)):
        return {"src": src, "status": "SKIP",
                "reason": f"not pandas ({type(obj).__name__})"}
    if not execute:
        shape = list(obj.shape) if isinstance(obj, pd.DataFrame) else [int(obj.shape[0])]
        return {"src": src, "dst": dst, "status": "DRY",
                "shape": shape,
                "type": type(obj).__name__, "engine": "pyarrow"}
    if isinstance(obj, pd.DataFrame):
        obj.to_parquet(dst, engine="pyarrow")
    else:  # Series
        obj.to_frame().to_parquet(dst, engine="pyarrow")
    obj2 = pd.read_parquet(dst, engine="pyarrow")
    # 用值相等（容忍 dtype 形式差异，如 StringDtype vs object）
    if isinstance(obj, pd.Series):
        obj_compare = obj.astype(str)
        obj2_compare = obj2.iloc[:, 0].astype(str)
        ok = len(obj) == len(obj2) and (obj_compare.values == obj2_compare.values).all()
        obj2_shape = [int(obj2.shape[0])]
    else:
        # DataFrame：所有列转 str 后逐列比对
        if obj.shape != obj2.shape:
            ok = False
        else:
            ok = True
            for col in obj.columns:
                if col not in obj2.columns:
                    ok = False; break
                if not (obj[col].astype(str).values == obj2[col].astype(str).values).all():
                    ok = False; break
        obj2_shape = list(obj2.shape)
    shape = list(obj.shape) if isinstance(obj, pd.DataFrame) else [int(obj.shape[0])]
    return {"src": src, "dst": dst, "status": "OK" if ok else "MISMATCH",
            "src_shape": shape, "dst_shape": obj2_shape,
            "src_md5": _md5(src), "dst_md5": _md5(dst),
            "src_size": os.path.getsize(src), "dst_size": os.path.getsize(dst)}


def migrate_tuple_pickle(src: str, execute: bool) -> dict:
    """tuple pickle（如 gse184869_symbol_matrix.pkl = (X, lab, os)）→ 多 .parquet + manifest。"""
    stem = src[:-4]
    obj = pickle.load(open(src, "rb"))
    if not isinstance(obj, tuple):
        return {"src": src, "status": "SKIP",
                "reason": f"expected tuple, got {type(obj).__name__}"}
    if len(obj) != 3:
        return {"src": src, "status": "SKIP",
                "reason": f"tuple length {len(obj)} != 3"}

    X, lab, os_data = obj
    # 类型校验：X 通常 DataFrame，lab Series，os_data DataFrame/Series
    pieces = [
        ("X", X, f"{stem}.X.parquet"),
        ("labels", lab, f"{stem}.labels.parquet"),
        ("os", os_data, f"{stem}.os.parquet"),
    ]
    plan = []
    for name, sub, path in pieces:
        if isinstance(sub, pd.DataFrame):
            plan.append({"name": name, "type": "DataFrame",
                         "dst": path, "shape": list(sub.shape)})
        elif isinstance(sub, pd.Series):
            plan.append({"name": name, "type": "Series",
                         "dst": path, "len": int(len(sub))})
        else:
            return {"src": src, "status": "SKIP",
                    "reason": f"tuple.{name} is {type(sub).__name__}, unsupported"}

    if not execute:
        return {"src": src, "pieces": plan, "status": "DRY",
                "manifest": f"{stem}.manifest.json"}

    # 真做
    for name, sub, path in pieces:
        if isinstance(sub, pd.DataFrame):
            sub.to_parquet(path, engine="pyarrow")
        elif isinstance(sub, pd.Series):
            # Series 无 to_parquet → 先 .to_frame()
            sub.to_frame().to_parquet(path, engine="pyarrow")
    manifest = {
        "src": src, "n_pieces": len(pieces),
        "pieces": [{"name": n, "path": p} for n, _, p in pieces],
        "format": "tuple(X=DataFrame, labels=Series, os=DataFrame|Series)",
    }
    with open(f"{stem}.manifest.json", "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 校验：再读回来对一遍（值相等，容忍 dtype 差异）
    X2 = pd.read_parquet(f"{stem}.X.parquet", engine="pyarrow")
    lab2 = pd.read_parquet(f"{stem}.labels.parquet", engine="pyarrow")
    os2 = pd.read_parquet(f"{stem}.os.parquet", engine="pyarrow")
    ok = (X2.shape == X.shape
          and len(lab2) == len(lab)
          and (X.values == X2.values).all()
          and (lab.astype(str).values == lab2.iloc[:, 0].astype(str).values).all()
          and os2.shape == os_data.shape
          and (os_data.iloc[:, 0].astype(str).values == os2.iloc[:, 0].astype(str).values).all())
    return {"src": src, "pieces": plan,
            "status": "OK" if ok else "MISMATCH",
            "src_size": os.path.getsize(src),
            "manifest": f"{stem}.manifest.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true",
                    help="真做（默认 dry-run）")
    ap.add_argument("--backup", action="store_true",
                    help="真做前把原 .pkl 拷到 /tmp/p1_backup_pkls/ 备份")
    args = ap.parse_args()

    print("=" * 78)
    print(f"00_pickle_to_parquet.py  ·  mode={'EXECUTE' if args.execute else 'DRY-RUN'}")
    print("=" * 78)

    pkls = sorted([os.path.join(AN, f) for f in os.listdir(AN) if f.endswith(".pkl")])
    if not pkls:
        print("(没有 .pkl 文件)")
        return

    if args.execute and args.backup:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        for p in pkls:
            shutil.copy2(p, BACKUP_DIR)
        print(f"已备份 {len(pkls)} 个 .pkl → {BACKUP_DIR}")

    report = []
    for src in pkls:
        name = os.path.basename(src)
        # 特殊处理：gse184869_symbol_matrix.pkl 是 tuple
        if name == "gse184869_symbol_matrix.pkl":
            r = migrate_tuple_pickle(src, args.execute)
        else:
            r = migrate_pandas_object(src, args.execute)
        report.append(r)
        status = r.get("status", "?")
        if status == "DRY":
            print(f"  [DRY] {name}: shape={r.get('shape') or r.get('pieces')}")
        elif status == "OK":
            sizes = f"src={r.get('src_size', '?')}B  dst={r.get('dst_size', '?')}B"
            print(f"  [OK]  {name}: {sizes}")
        elif status == "MISMATCH":
            print(f"  [❌ MISMATCH] {name}: {r}")
        elif status == "SKIP":
            print(f"  [SKIP] {name}: {r.get('reason')}")

    # 写报告
    if not args.execute:
        with open(REPORT_PATH, "w") as f:
            json.dump({"mode": "DRY-RUN", "n_pickles": len(pkls),
                       "items": report}, f, ensure_ascii=False, indent=2)
        print(f"\n报告已写：{REPORT_PATH}")
        print("→ 真做请加 --execute（建议同时 --backup）")
    else:
        with open(REPORT_PATH, "w") as f:
            json.dump({"mode": "EXECUTE", "n_pickles": len(pkls),
                       "items": report}, f, ensure_ascii=False, indent=2)
        ok = sum(1 for r in report if r.get("status") == "OK")
        print(f"\n{'=' * 78}")
        print(f"完成 {ok}/{len(report)} 迁移")
        print(f"报告：{REPORT_PATH}")


if __name__ == "__main__":
    main()