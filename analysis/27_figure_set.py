"""analysis/27_figure_set.py — 原全图件统一生成脚本

⚠️ 状态说明：2026-09-14 在批量修改面板标签大小写时，本文件被误操作截断为 0 字节。
后续 8 张主图/补充图已在 40_figure_QA_redraw.py 中按 nature-figure QA 契约重绘完成。

本文件现作为兼容层保留，实际生成逻辑转发到 40_figure_QA_redraw.py。
Figure5_mcp_counter 未在重绘范围（原图审计 PASS），仍沿用 01_投稿资料包_BiB/figures 下的原稿。
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REDIRECT = os.path.join(_HERE, "40_figure_QA_redraw.py")


def _load_redraw():
    spec = importlib.util.spec_from_file_location("_redraw40", _REDIRECT)
    m = importlib.util.module_from_spec(spec)
    # 保证重绘模块能拿到 PYTHONPATH 中的对齐门控脚本
    skill_scripts = os.path.expanduser("~/.workbuddy/skills/nature-figure/scripts")
    for p in (_HERE, skill_scripts):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec.loader.exec_module(m)
    return m


def figure1():
    """稿件 Fig 1 — audit pipeline"""
    _load_redraw().fig1_pipeline()


def figure2():
    """稿件 Fig 2 — null / CV / discreteness"""
    _load_redraw().fig2_null_cv()


def figure3():
    """稿件 Fig 3 — histology confound"""
    _load_redraw().fig3_histology()


def figure4():
    """稿件 Fig 4 — cross-cohort consistency"""
    _load_redraw().fig4_cross_cohort()


def figure6():
    """稿件 Fig 6 — L5 survival + PDO"""
    _load_redraw().fig6_L5()


def supp_s1():
    """Suppl Fig S1 — programme scores"""
    _load_redraw().supp_s1_scores()


def supp_s2():
    """Suppl Fig S2 — ABIS crosscheck"""
    _load_redraw().supp_s2_abis()


def supp_s3():
    """Suppl Fig S3 — purity correction"""
    _load_redraw().supp_s3_purity()


def main():
    m = _load_redraw()
    m.fig1_pipeline()
    m.fig2_null_cv()
    m.fig3_histology()
    m.fig4_cross_cohort()
    m.fig6_L5()
    m.supp_s1_scores()
    m.supp_s2_abis()
    m.supp_s3_purity()
    print("\n注：Figure5_mcp_counter 保持原稿，未重绘。")


if __name__ == "__main__":
    main()
