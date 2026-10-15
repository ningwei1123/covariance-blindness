# -*- coding: utf-8 -*-
"""V99：复 Ginzburg-Landau 的多先验复检 —— 重点是**换到混沌参数区**。

发现的错误（与 SH 同类）
  我们此前测的是 (b, c) = (0.5, 5.0) 与 (2.0, 2.0)，而 CGLE
      d_t A = A + (1 + i b) d_x^2 A - (1 + i c) |A|^2 A
  的均匀态稳定判据是 **Benjamin-Feir / Newell：1 + b c > 0 稳定**，`1 + b c < 0` 不稳定。
      (0.5, 5.0): b c = +2.5  -> 1 + b c = 3.5 > 0  **稳定（非混沌）**
      (2.0, 2.0): b c = +4.0  -> 1 + b c = 5.0 > 0  **稳定（非混沌）**
  ⇒ 两次测试都在**非混沌区**，故"无肘部"是必然的，不是估计器的问题。

本脚本
  在 BF **不稳定**区（b c < -1）扫描，先判混沌（d_eff > 5），再跑判据（五个先验族）。
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"E:\nature投稿\探索1\数值验证")
from v54b_screen_families import cgle_run          # noqa: E402
from v95_multi_prior import FAMILIES, margin_scan  # noqa: E402
from v20_robust import elbow                       # noqa: E402
from v34_audit import sig_profile, shuffle_tail    # noqa: E402

OUT = {}


def to_state(A):
    """复场 -> 实状态向量 [Re A, Im A]。"""
    return np.hstack([A.real, A.imag])


def main():
    print("=" * 116)
    print("V99  CGLE 多先验复检（重点：BF 不稳定区 bc < -1）")
    print("=" * 116)
    print(f"  {'b':>5}{'c':>6}{'bc':>7}{'1+bc':>7}{'L':>7}{'d_eff':>9}"
          f"{'最佳margin':>12}{'先验':>17}{'肘部':>6}{'s_orig':>8}{'s_shuf':>8}  判读")
    plan = []
    # BF 稳定区（此前用过的，作对照）
    plan += [(0.5, 5.0, 300.0), (2.0, 2.0, 200.0)]
    # BF 不稳定区（应有相/缺陷湍流）
    for b, c in ((2.0, -2.0), (3.0, -2.0), (2.0, -3.0), (3.0, -3.0), (1.5, -2.0)):
        for L in (100.0, 200.0):
            plan.append((b, c, L))
    for b, c, L in plan:
        bc = b * c
        try:
            X = cgle_run(b=b, c=c, L=L, N=512, dt=0.02, T=250.0, warm=150.0,
                            stride=4, seed=3)
        except Exception as e:
            print(f"  {b:>5}{c:>6}{bc:>7.1f}{1+bc:>7.1f}{L:>7.0f}   运行失败 "
                  f"{type(e).__name__}")
            continue
        X = np.asarray(X, dtype=np.float64)
        if len(X) > 3000:
            X = X[:: max(1, len(X) // 3000)]
        s = np.linalg.svd(X - X.mean(0), compute_uv=False)
        lam = s ** 2 / (s ** 2).sum()
        deff = float(1 / (lam ** 2).sum())
        ds = list(range(2, min(30, X.shape[1] - 1)))
        best = (-1e30, None, None, None); bfam = None
        for fam in FAMILIES:
            try:
                mg, d, sm, rm = margin_scan(X, fam, ds, excl=20)
            except Exception:
                continue
            if mg > best[0]:
                best = (mg, d, sm, rm); bfam = fam
        mg, d, sm, rm = best
        try:
            res, el = elbow(X, ds, deg=2, nval=600)
            so = sig_profile({dd: res[dd]["ratio"] for dd in ds}, ds)
            de = el[0]
            Xs, _ = shuffle_tail(X, de, seed=0)
            r2, _ = elbow(Xs, ds, deg=2, nval=600)
            s2 = sig_profile({dd: r2[dd]["ratio"] for dd in ds}, ds)
            s_o, s_s = float(so.get(de, np.nan)), float(s2.get(de, np.nan))
        except Exception:
            de, s_o, s_s = None, float("nan"), float("nan")
        P0 = deff > 5.0
        P3 = (s_s > 0.9) and (s_o < 0.7)
        tag = f"b{b}_c{c}_L{L:.0f}"
        print(f"  {b:>5}{c:>6}{bc:>7.1f}{1+bc:>7.1f}{L:>7.0f}{deff:>9.2f}"
              f"{mg:>12.4f}{str(bfam):>17}{str(de):>6}{s_o:>8.3f}{s_s:>8.3f}  "
              f"{'**A 级候选**' if (P0 and mg > 0 and P3) else ('混沌' if P0 else '非混沌')}"
              f"{'' if mg > 0 else ' + 判据拒'}")
        OUT[tag] = dict(b=b, c=c, bc=bc, BF_stable=bool(1 + bc > 0), L=L,
                        d_eff=deff, margin=mg, family=bfam, d=d,
                        elbow=de, s_orig=s_o, s_shuf=s_s,
                        P0=bool(P0), P3=bool(P3))
    json.dump(OUT, open(os.path.join(HERE, "v99_cgle_multiprior.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print("\n  判读：")
    print("    · BF 稳定区（1+bc>0）应给很小的 d_eff —— 那是**非混沌**，不是估计器的问题；")
    print("    · BF 不稳定区（1+bc<0）若 d_eff 显著 > 1 而 margin 仍为负，则负面在**混沌区**成立。")
    print("\nsaved v99_cgle_multiprior.json")


if __name__ == "__main__":
    main()
