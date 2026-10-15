# -*- coding: utf-8 -*-
"""V96：Swift-Hohenberg 的**大长宽比**重测 —— 负面是参数范围问题还是系统问题？

V95 的诊断
  SH r=1.5 c=16 的 d_eff = **1.00**，r=2.0 c=24 为 1.03 ⇒ 数据本身是**秩 1 的静止图案**，
  根本不是混沌。故 SH 的负面是**参数范围**的陈述，不是系统陈述。
  文献（Greenside & Cross 等）记载 SH 的时空混沌出现在**更大长宽比**。

本脚本
  在 Γ = Lx/(2 pi) = 16 / 32 / 48 / 64 / 96 上重跑 SH，报告
    · d_eff（判断是否真的混沌）
    · 判据 margin（五个先验族中的最佳）
  若大 Γ 下 d_eff 显著大于 1 且判据转正，则 SH 负面须撤回。
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v25_sh                                   # noqa: E402
from v95_multi_prior import FAMILIES, margin_scan  # noqa: E402

OUT = {}


def main():
    print("=" * 108)
    print("V96  Swift-Hohenberg 的大长宽比重测")
    print("=" * 108)
    print(f"  {'Gamma':>7}{'r':>5}{'N':>5}{'n_unstable':>11}{'d_eff':>9}"
          f"{'最佳margin':>12}{'先验':>18}{'于 d=':>7}  判读")
    for Gamma, r in ((48, 2.0), (48, 4.0), (48, 8.0), (48, 16.0),
                     (64, 8.0), (64, 16.0), (96, 8.0)):
        Lx = 2 * np.pi * Gamma
        N = int(min(1024, max(256, 8 * Gamma)))
        try:
            X, k, ku = v25_sh.sh_run(N=N, Lx=Lx, r=r, dt=0.05, T=800.0,
                                     stride=6, warm=400.0, seed=3)
        except Exception as e:
            print(f"  {Gamma:>7}{r:>5}{N:>5}   运行失败 {type(e).__name__}")
            continue
        if X.ndim > 2:
            X = X.reshape(len(X), -1)
        X = np.asarray(X, dtype=np.float64)
        if len(X) > 3000:
            X = X[:: max(1, len(X) // 3000)]
        s = np.linalg.svd(X - X.mean(0), compute_uv=False)
        lam = s ** 2 / (s ** 2).sum()
        deff = float(1 / (lam ** 2).sum())
        ds = list(range(2, min(30, X.shape[1] - 1)))
        best = (-1e30, None, None, None)
        bfam = None
        for fam in FAMILIES:
            try:
                mg, d, sm, rm = margin_scan(X, fam, ds, excl=20)
            except Exception:
                continue
            if mg > best[0]:
                best = (mg, d, sm, rm); bfam = fam
        mg, d, sm, rm = best
        chaotic = deff > 5.0
        print(f"  {Gamma:>7}{r:>5}{N:>5}{ku:>11}{deff:>9.2f}"
              f"{mg:>12.4f}{str(bfam):>18}{str(d):>7}  "
              f"{'混沌' if chaotic else '非混沌'} + "
              f"{'**认证**' if mg > 0 else '拒绝'}")
        OUT[f"G{Gamma}_r{r}"] = dict(Gamma=Gamma, r=r, N=N, n_unstable=int(ku),
                                     d_eff=deff, margin=mg, family=bfam, d=d,
                                     sigma_min=sm, R_max=rm, chaotic=bool(chaotic))
    json.dump(OUT, open(os.path.join(HERE, "v96_sh_large_gamma.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print("\n  判读：若某些 (Gamma, r) 的 d_eff 显著 > 1 而 margin 仍为负，")
    print("        则 SH 的负面在**混沌参数区**依然成立；若 margin 转正，则须撤回。")
    print("\nsaved v96_sh_large_gamma.json")


if __name__ == "__main__":
    main()
