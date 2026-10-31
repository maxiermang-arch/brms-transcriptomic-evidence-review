#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 3 · 步骤 15（修订 4）：mygene.info 批量取基因长度 → 真 TPM（ABIS 官方要求 TPM 输入）
议题 A · 2026-09-13

迭代记录（每一次失败都写下来，避免重犯）：
 v1  mygene.info `genomic_pos`，用 GET 逐基因/小批 → 覆盖率仅 1585/25188（6%）
     ✗ 且脚本用中位数静默填充 → 产出"假 TPM"（= CPM），无任何报错
 v2  Ensembl REST `POST /lookup/symbol?expand=1` → 服务端 HTTP 500（批量接口不支持 expand）
 v3  Ensembl REST `POST /lookup/symbol`（无 expand，基因跨度 end−start+1）
     ⚠️ 写出 TPM 但 2026-09-13 复核发现**反推长度为常数 27,000 bp（= 中位数）**
        → 说明长度查询整体失败、又被中位数填充，TPM ≡ CPM。**静默降级第二次发生**
 v4（本版）改用 **mygene.info `POST /v3/query`**（scopes=symbol, species=human,
     fields=symbol,genomic_pos），批 500
     · 实测覆盖率 86.7%（1734/2000），2000 基因 8.6 s → 全量约 100 s
     · **新增覆盖率硬断言**：覆盖 < 85% 直接 raise，绝不再静默中位数填充
     · 长度 = 基因跨度 |end−start|+1；多重定位取主染色体（1-22,X,Y）记录
     · 地板 300 bp（避免个别异常小区间）

⚠️ 已知局限（写入正文）：长度用**基因跨度**（含内含子），而 TPM 的严格定义是
   用**外显子并集/成熟转录本长度**。对超长内含子基因（TTN 305 kb、CNTNAP2 2.3 Mb）
   会系统性高估长度、低估其 TPM。mygene 的 `exons` 字段经实测不可用（返回 1–2 条
   结构异常记录），故无法修正。影响评估：ABIS 是基因层线性模型，长度权重偏差
   主要影响**个别超长基因**，对 17 类免疫细胞相对构成的影响有限——但绝对尺度
   （是否恰好 0–100）不应过度解读。
"""
import json, os, time, pickle, numpy as np, pandas as pd, requests, warnings
# —— 共享配置（P0-2 去硬编码路径）——
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, AN, PU, SUBS, safe_makedirs  , load_pkl_tuple  , load_dataframe  # noqa
warnings.filterwarnings("ignore")

AN, PU = f"{ROOT}/analysis", f"{ROOT}/data/purity"
LENC = f"{PU}/gene_lengths_mygene.tsv"
CACHEJ = f"{PU}/gene_lengths_lookup_v4.json"
URL = "https://mygene.info/v3/query"
PRIMARY = {str(i) for i in range(1, 23)} | {"X", "Y"}


def load(which):
    """返回 (表达矩阵, 标签)，矩阵行=基因、列=样本；队列 B 从 log2 还原线性。"""
    if which == "A":
        X = load_dataframe(f"{AN}/X_counts.pkl")
        lab = load_dataframe(f"{AN}/labels_gse245467.pkl")
        k = lab[lab.isin(SUBS)].index
        return X[k], lab[k]
    X, lab, _ = load_pkl_tuple(f"{AN}/gse184869_symbol_matrix.pkl")
    k = lab[lab.isin(SUBS)].index
    return np.power(2.0, X[k]) - 1.0, lab[k]


def pick_pos(gp):
    """genomic_pos 可能是 dict 或 list（多定位/多版本）→ 优先主染色体。"""
    if isinstance(gp, list):
        prim = [d for d in gp if isinstance(d, dict) and str(d.get("chr", "")).replace("chr", "") in PRIMARY]
        gp = (prim or gp or [None])[0]
    return gp


def fetch_lengths(genes):
    """批量取基因跨度；带覆盖率硬断言，禁止静默降级。"""
    got = {}
    if os.path.exists(CACHEJ) and os.path.getsize(CACHEJ) > 2:
        got = json.load(open(CACHEJ))
    todo = [g for g in genes if g not in got]
    print(f"缓存 {len(got)}  待查 {len(todo)}", flush=True)

    B, ok, bad = 500, 0, 0
    t0 = time.time()
    for i in range(0, len(todo), B):
        ch = todo[i:i + B]
        for a in range(3):
            try:
                r = requests.post(
                    URL,
                    data={"q": ",".join(ch), "scopes": "symbol", "species": "human",
                          "fields": "symbol,genomic_pos,type_of_gene"},
                    timeout=45)
                if r.status_code == 200:
                    for h in r.json():
                        q = h.get("query")
                        gp = pick_pos(h.get("genomic_pos"))
                        if isinstance(gp, dict) and gp.get("start") and gp.get("end"):
                            got[q] = {"span": abs(int(gp["end"]) - int(gp["start"])) + 1,
                                      "chr": gp.get("chr"), "type": h.get("type_of_gene")}
                    ok += 1
                    break
                bad += 1
                time.sleep(1.0 * (a + 1))
            except Exception:
                bad += 1
                time.sleep(1.0 * (a + 1))
        if (i // B) % 10 == 0:
            print(f"  {min(i+B,len(todo))}/{len(todo)} 累计={len(got)}  {time.time()-t0:.0f}s", flush=True)

    json.dump(got, open(CACHEJ, "w"))
    cov = len([g for g in genes if g in got]) / len(genes)
    print(f"批次 ok={ok} bad={bad}  覆盖 {len(got)}/{len(genes)} = {cov:.1%}  用时 {time.time()-t0:.0f}s")
    assert cov >= 0.85, f"❌ 基因长度覆盖率仅 {cov:.1%} < 85% —— 拒绝生成 TPM（防止再次静默降级）"
    return got



def main():
    XA, _ = load("A")
    XB, _ = load("B")
    genes = sorted(set(XA.index) | set(XB.index))
    print(f"合并基因集: {len(genes)}")
    
    got = fetch_lengths(genes)
    
    rows = []
    for g in genes:
        o = got.get(g)
        span = o["span"] if isinstance(o, dict) else np.nan
        rows.append({"gene": g, "length": span,
                     "chr": (o.get("chr") if isinstance(o, dict) else None),
                     "type": (o.get("type") if isinstance(o, dict) else None)})
    L = pd.DataFrame(rows).set_index("gene")
    L["length"] = L["length"].clip(lower=300)          # 地板 300 bp
    miss = int(L["length"].isna().sum())
    med = float(L["length"].median())
    L["length"] = L["length"].fillna(med)
    L.to_csv(LENC, sep="\t")
    print(f"取得长度 {len(L)-miss}/{len(L)}（缺失 {miss} 用中位 {med:.0f} bp 填充）")
    v = L["length"].values
    print("长度分位 bp:", {f"p{int(q*100)}": int(np.quantile(v, q)) for q in (.05, .25, .5, .75, .95, .99)})
    for g2 in ("TTN", "CNTNAP2", "ACTB", "TP53", "CD3D", "EGFR"):
        if g2 in L.index:
            print(f"  {g2}: {int(L.loc[g2,'length']):,} bp")
    
    for which in ("A", "B"):
        X, lab = load(which)
        l = L["length"].reindex(X.index).values.reshape(-1, 1)
        rate = X.values / l
        tpm = rate / rate.sum(axis=0, keepdims=True) * 1e6
        pd.DataFrame(tpm, index=X.index, columns=X.columns).to_csv(f"{PU}/cohort{which}_TPM.tsv", sep="\t")
        sig = pd.read_csv(f"{PU}/sigmatrixRNAseq.txt", sep="\t", index_col=0)
        _s = np.asarray(tpm.sum(axis=0)).ravel()
        print(f"队列 {which}: {tpm.shape}  列和中位={np.median(_s):.0f} (min={_s.min():.0f}, max={_s.max():.0f})  "
              f"ABIS 覆盖={len(set(sig.index) & set(X.index))}/{sig.shape[0]}")
    print("✅ 真 TPM 导出完成")
    

if __name__ == "__main__":
    main()
