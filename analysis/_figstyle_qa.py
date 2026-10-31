"""analysis/_figstyle_qa.py — 投稿图件统一加固样式（2026-09-14）

背景
----
nature-figure 技能对 01_投稿资料包_BiB/figures 下 9 张图做过审计，发现三类系统性问题：

1. **版心缩放导致字号腰斩**：Fig7 画布 391 mm、Fig8 画布 287 mm。
   放到期刊 174 mm 双栏版心后，纸面等效字号 = 图内字号 × 174 ÷ 画布宽度，
   8 pt → 3.56 pt、7.4 pt → 4.49 pt，均低于 5 pt 硬下限。
2. **mathtext 收缩**：对数轴刻度走 mathtext 渲染，上标缩到父字号的 ~0.7，
   6.5 pt → 4.55 pt（Figure2 面板 C）。
3. **文字压线 / 文字互压**：fig.text 置于轴下方（负 y）、面板标签用 axes-fraction
   1.08、图例压在柱顶、数值标签贴着柱顶边线。

本模块把修正固化下来，供 40_figure_QA_redraw.py 及后续图件复用。

硬约束
------
* 画布宽恒为 174 mm（BiB 双栏全宽），**不用 bbox_inches="tight"**——
  tight 会裁掉空白使画布变窄，虽不会让字变小，但会让不同图件的实际版心宽度
  互不相同（149–179 mm 不等），无法统一核算等效字号。
* 最小渲染字号 7.0 pt（> 5 pt 技能下限，> 6 pt 实际可读下限）。
* 禁用 mathtext：刻度/标注一律用 Unicode（ρ ≥ → × ²）。
* 面板标签用**固定点偏移**（offset points），不用 axes fraction（高矮面板
  在同一 fraction 下物理位移不同，会对不齐）。
* 不放任何 fig.text 在轴下方；注释并入 xlabel / title 的换行第二行，
  保证是单个文本对象，物理上不可能自重叠。
"""
from __future__ import annotations

import os
import sys

import matplotlib as mpl

# 让本目录的 audit_panel_alignment 可导入（技能脚本目录也加入 PYTHONPATH）
_HERE = os.path.dirname(os.path.abspath(__file__))
_BUNDLED_TOOLS = os.path.join(os.path.dirname(_HERE), 'tools')
for _p in (_HERE, _BUNDLED_TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MM = 1 / 25.4
FULL_W_MM = 174.0          # 双栏全宽（画布宽度，非版心缩放宽度）
FULL_W = FULL_W_MM * MM    # 英寸

# ---- 字号：最小 7.0 pt ----
FS_TITLE = 8.5
FS_LABEL = 7.5
FS_TICK = 7.0
FS_ANNOT = 7.0
FS_LEGEND = 7.0
FS_PANEL = 10.0

# ---- 配色（沿用 27_figure_set.py，全稿一致）----
C_A = "#2E5F8A"        # 蓝：score only / in-sample
C_B = "#C0392B"        # 红：+ histology / CV
C_G = "#7F8C8D"        # 灰：null
C_OK = "#1E8449"       # 绿
C_WARN = "#B9770E"     # 橙
SUB_COLOR = {"1": "#4C72B0", "2": "#DD8452", "3": "#55A868", "4": "#C44E52"}
COH_COLOR = {"A": "#4C72B0", "B": "#DD8452"}

mpl.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FS_LABEL,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_TITLE,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 14,          # 给面板标签与标题留垂直间隙
    "legend.frameon": False,
    "svg.fonttype": "none",       # SVG 文字可编辑
    "pdf.fonttype": 42,           # PDF 内嵌 TrueType，文字可编辑
    "ps.fonttype": 42,
})


def panel_tag(ax, tag, dx_pt=-19, dy_pt=7):
    """面板标签：小写粗体，固定点偏移，挂在轴左上角外侧。

    用小写 a/b/c —— nature-figure 对齐门控只识别 `[a-z]` 粗体左上角标签，
    且 Nature 体系图件规范亦为小写粗体。

    用 offset points 而非 axes fraction —— 后者在高矮不一的面板上
    会产生不同的物理位移，导致同一行面板标签的 y 锚点对不齐。
    """
    ax.annotate(
        tag, xy=(0, 1), xycoords="axes fraction",
        xytext=(dx_pt, dy_pt), textcoords="offset points",
        fontsize=FS_PANEL, fontweight="bold",
        ha="left", va="bottom", annotation_clip=False,
    )


def save_qa(fig, name, outdir, *, axes=None, multipanel=True, exclude_axes=(),
            row_groups=None, column_groups=None, panel_ids=None,
            dpi=600, strict_alignment=True):
    """导出并跑强制门控。

    多面板图：先跑 render-time 对齐门控（容差 1.5 pt），失败即抛异常，
    不导出交付包。再导出 PDF（矢量）+ SVG + PNG。
    """
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.join(outdir, name)

    if multipanel:
        import json
        from audit_panel_alignment import require_matplotlib_panel_alignment
        # 注意：skill 脚本写 json/svg 用的是「先写 .tmp 再 unlink 替换」的原子写，
        # 在本机受保护目录（Desktop）下 unlink 会被系统拒绝（PermissionError）。
        # 因此这里让门控只返回结果，报告由本函数直接覆盖写（不做 unlink）。
        report = require_matplotlib_panel_alignment(
            fig,
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=True,
            strict=strict_alignment,
            axes=axes,
            exclude_axes=list(exclude_axes),
            row_groups=row_groups,
            column_groups=column_groups,
            panel_ids=panel_ids,
        )
        with open(f"{stem}.alignment.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"    alignment: {report.get('verdict')}")

    fig.savefig(f"{stem}.pdf", facecolor="white")
    fig.savefig(f"{stem}.svg", facecolor="white")
    fig.savefig(f"{stem}.png", dpi=dpi, facecolor="white")
    print(f"    → {name}.pdf / .svg / .png ({dpi} dpi)")


def plain_log_formatter():
    """对数轴刻度用纯文本，绕开 mathtext 0.7 收缩。"""
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, pos: f"{v:g}")
