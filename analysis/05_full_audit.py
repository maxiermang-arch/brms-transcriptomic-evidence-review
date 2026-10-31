#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
阶段 2 · 步骤 5：四亚型定义维度完整审计
议题 A · 2026-09-13

把论文四个亚型的"定义维度"逐一拿出来，用**独立基因集 + 独立打分方法**在**独立公开数据集**上
检验其可复现性。这是本议题的核心产出。

  BrMS1 neural-like      ← 神经发育通路（GO BP）
  BrMS2 immune + EMT     ← T 细胞活化 / IFN-γ / EMT / 免疫排斥
  BrMS3 metabolic        ← OXPHOS / FAO / Glycolysis
  BrMS4 proliferative    ← E2F / G2-M
"""
import json, warnings, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# —— 共享配置（P0-2 去硬编码路径）——
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, AN, FG, SEED  , load_dataframe  # noqa
warnings.filterwarnings("ignore")


def main():
    plt.rcParams.update({"font.size":8,"axes.linewidth":0.7,"font.family":"DejaVu Sans"})
    
    X=load_dataframe(f"{AN}/X_counts.pkl")
    lab=load_dataframe(f"{AN}/labels_all.pkl")["brms"]
    lab=lab[lab.index.isin(X.columns)]; lab=lab[lab.isin(list("1234"))]
    Xs=X[lab.index]
    hall=json.load(open(f"{AN}/genesets/hallmark.json"))
    neur=json.load(open(f"{AN}/genesets/neural.json"))
    L=np.log2(Xs.div(Xs.sum(axis=0),axis=1)*1e6+1)
    Z=L.sub(L.mean(axis=1),axis=0).div(L.std(axis=1).replace(0,np.nan),axis=0).dropna(how="all")
    
    # ---- 定义四个"维度"的基因集
    DIMS = {
     "Neural-like\n(BrMS1)": [("GO",k,v) for k,v in neur.items()],
     "Immune / EMT\n(BrMS2)": [("HM","Interferon Gamma Response",hall["Interferon Gamma Response"]),
                               ("HM","Epithelial Mesenchymal Transition",hall["Epithelial Mesenchymal Transition"]),
                               ("HM","Allograft Rejection",hall["Allograft Rejection"]),
                               ("HM","Inflammatory Response",hall["Inflammatory Response"])],
     "Metabolic\n(BrMS3)": [("HM","Oxidative Phosphorylation",hall["Oxidative Phosphorylation"]),
                            ("HM","Fatty Acid Metabolism",hall["Fatty Acid Metabolism"]),
                            ("HM","Glycolysis",hall["Glycolysis"]),
                            ("HM","mTORC1 Signaling",hall["mTORC1 Signaling"])],
     "Proliferative\n(BrMS4)": [("HM","E2F Targets",hall["E2F Targets"]),
                                ("HM","G2-M Checkpoint",hall["G2-M Checkpoint"]),
                                ("HM","Myc Targets V1",hall["Myc Targets V1"])],
    }
    SUB_OF = {"Neural-like\n(BrMS1)":"1","Immune / EMT\n(BrMS2)":"2",
              "Metabolic\n(BrMS3)":"3","Proliferative\n(BrMS4)":"4"}
    
    print("="*80); print("四亚型定义维度审计（独立基因集 + 独立方法 + 独立数据）"); print("="*80)
    rows=[]; perm_rows=[]
    for dim, items in DIMS.items():
        sub=SUB_OF[dim]; y=(lab==sub).astype(int).values
        print(f"\n### {dim.replace(chr(10),' ')}   (n={(lab==sub).sum()} vs {(lab!=sub).sum()})")
        print(f"  {'基因集':<42}{'覆盖':>6}{'AUC':>8}{'Cliff δ':>9}{'p(MWU)':>9}{'p(perm)':>9}")
        for src,name,glist in items:
            g=[x for x in glist if x in Z.index]
            if len(g)<10: print(f"  {name[:40]:<42} 覆盖不足"); continue
            s=Z.loc[g].mean(axis=0)
            auc=roc_auc_score(y,s.values)
            a=s[lab==sub].values; b=s[lab!=sub].values
            gt=np.sum(a[:,None]>b[None,:]); lt=np.sum(a[:,None]<b[None,:])
            cliff=(gt-lt)/(len(a)*len(b))
            pv=stats.mannwhitneyu(a,b,alternative="two-sided").pvalue
            # 置换
            cnt=0; NB=1000
            for _ in range(NB):
                yp=RNG.permutation(y)
                if roc_auc_score(yp,s.values)>=auc: cnt+=1
            pp=(cnt+1)/(NB+1)
            star="***" if pp<0.001 else "**" if pp<0.01 else "*" if pp<0.05 else "ns"
            print(f"  {name[:40]:<42}{len(g):>6}{auc:>8.3f}{cliff:>9.3f}{pv:>9.4f}{pp:>9.4f} {star}")
            rows.append({"dimension":dim.replace("\n"," "),"geneset":name,"n_genes":len(g),
                         "auc":auc,"cliff_delta":cliff,"p_mwu":pv,"p_perm":pp,
                         "target_subtype":sub})
    res=pd.DataFrame(rows)
    res.to_csv(f"{AN}/audit_four_subtypes.csv",index=False)
    
    # ---- 汇总：每个维度的"最佳 AUC"
    print("\n"+"="*80); print("★ 汇总：每个亚型的定义维度能否在独立数据复现"); print("="*80)
    summ=res.groupby("dimension").agg(n_genesets=("geneset","size"), best_auc=("auc","max"),
                                       mean_auc=("auc","mean"), min_p=("p_perm","min"))
    summ["verdict"]=["✅ 可复现" if r.best_auc>=0.75 and r.min_p<0.01 else
                     "🟡 弱/方法依赖" if r.best_auc>=0.62 else "❌ 不可复现"
                     for r in summ.itertuples()]
    print(summ.round(3).to_string())
    summ.to_csv(f"{AN}/audit_summary.csv")
    
    # ================================ 图 4：审计总图 ================================
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),dpi=400,
                          gridspec_kw={"width_ratios":[1.35,1]})
    ax=axes[0]
    order=["Neural-like (BrMS1)","Immune / EMT (BrMS2)","Metabolic (BrMS3)","Proliferative (BrMS4)"]
    cols={"Neural-like (BrMS1)":"#7d3c98","Immune / EMT (BrMS2)":"#1e8449",
          "Metabolic (BrMS3)":"#c0392b","Proliferative (BrMS4)":"#2471a3"}
    ypos=0; ticks=[]; tlabels=[]
    for dim in order:
        sub=res[res.dimension==dim]
        for _,r in sub.iterrows():
            c=cols[dim]
            ax.barh(ypos, r.auc, height=0.72, color=c, alpha=0.85, edgecolor="k", linewidth=0.4)
            ax.text(r.auc+0.008, ypos, f"{r.auc:.2f}", va="center", fontsize=5.6)
            if r.p_perm>=0.05:
                ax.text(min(r.auc,0.72)-0.015, ypos, "n.s.", va="center", ha="right",
                        fontsize=6.0, color="#7b241c", fontweight="bold")
            ticks.append(ypos); tlabels.append(r.geneset[:34]); ypos+=1
        ypos+=0.55
    ax.axvline(0.5,color="gray",ls="--",lw=0.9)
    ax.axvline(0.75,color="green",ls=":",lw=0.9)
    ax.set_yticks(ticks); ax.set_yticklabels(tlabels,fontsize=5.8)
    ax.invert_yaxis(); ax.set_xlim(0.2,1.0)
    ax.set_xlabel("AUC (one-vs-rest)",fontsize=7.5)
    ax.set_title("Replicability of BrMS subtype definitions",fontsize=8,pad=5)
    ax.text(0.755,ax.get_ylim()[0]+0.3,"AUC 0.75",fontsize=5.4,color="green",rotation=90,va="top")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    
    ax2=axes[1]
    sm=summ.loc[order]
    bcol=["#7d3c98","#1e8449","#c0392b","#2471a3"]
    x=np.arange(4)
    ax2.bar(x,sm["best_auc"],0.62,color=bcol,alpha=0.9,edgecolor="k",linewidth=0.5)
    for xi,v in zip(x,sm["best_auc"]): ax2.text(xi,v+0.015,f"{v:.2f}",ha="center",fontsize=6.5)
    ax2.axhline(0.5,color="gray",ls="--",lw=0.9)
    ax2.axhline(0.75,color="green",ls=":",lw=0.9)
    ax2.set_xticks(x); ax2.set_xticklabels(["BrMS1\nneural","BrMS2\nimmune/EMT","BrMS3\nmetabolic","BrMS4\nprolifer."],fontsize=6.2)
    ax2.set_ylim(0.4,1.0); ax2.set_ylabel("Best AUC within dimension",fontsize=7)
    ax2.set_title("Best replicating signal per subtype",fontsize=8,pad=5)
    for xi,v,vd in zip(x,sm["best_auc"],
                       ["replicated" if v>=0.75 else ("weak" if v>=0.62 else "not repl.") for v in sm["best_auc"]]):
        ax2.text(xi,0.425,vd,ha="center",fontsize=5.8,
                 color="#1e8449" if vd=="replicated" else ("#b9770e" if vd=="weak" else "#a93226"),
                 fontweight="bold")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig4_subtype_audit.png",bbox_inches="tight")
    fig.savefig(f"{FIG}/Fig4_subtype_audit.pdf",bbox_inches="tight")
    print(f"\n  ✅ Fig4 → figures/Fig4_subtype_audit.png/.pdf")
    print("\n  ✅ 已存 audit_four_subtypes.csv / audit_summary.csv")
    

if __name__ == "__main__":
    main()
