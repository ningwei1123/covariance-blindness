# -*- coding: utf-8 -*-
"""V98：**低 Re 的 2D Kolmogorov 重测** —— 唯一"文献预测小维数、而我们只测了高维区制"的负面。

文献依据（Cleary & Page 2025，本机全文）
  · 2D Kolmogorov，强迫波数 **n = 4**（与我们求解器一致：omega_t + J = nu*Delta - 4cos(4y)）；
  · 25 <= Re <= 400，结果 **d_A ~ Re^{1/3}** —— 比全局吸引子界 Re^{4/3} 弱得多；
  · 并明确写道："Linear approaches, such as **principal component analysis (PCA)** and its
    extensions ... **tend to overestimate the dimension as they are limited to linear subspaces**"
    —— **与我们"r(eps) 高估 2.3-2.6 倍"的结论直接吻合**。
  · 另有："The Lyapunov dimension ... depends strongly on the **wavenumber n** ... and only
    weakly on Re"。

我们的缺口
  只测过 nu = 0.02（Re = 50，d_eff = 19.79，**高维**）。而按 d_A ~ Re^{1/3}，
  **Re = 25 时 d_A ≈ 3** —— 低维区从未测过。

本脚本
  在 Re = 1/nu = 20 / 25 / 33 / 40 / 50 上重跑，对每个参数计算
    · d_eff（是否混沌 —— P0 前提）
    · 判据 margin（五个先验族取最佳）
    · 检测器肘部与尾部打乱（P3）
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"E:\nature投稿\探索1\数值验证")
from b2_ns2d import run_resolution            # noqa: E402
from v95_multi_prior import FAMILIES, margin_scan   # noqa: E402
from v20_robust import elbow                  # noqa: E402
from v34_audit import sig_profile, shuffle_tail    # noqa: E402

OUT = {}


def main():
    print("=" * 112)
    print("V98  低 Re 的 2D Kolmogorov 重测（强迫 n = 4，与 Cleary & Page 一致）")
    print("=" * 112)
    print(f"  {'Re':>5}{'nu':>7}{'N':>5}{'样本':>7}{'d_eff':>8}{'能量涨落%':>11}"
          f"{'最佳margin':>12}{'先验':>18}{'肘部':>6}{'s_orig':>8}{'s_shuf':>8}  判读")
    for nu in (0.05, 0.04, 0.03, 0.025, 0.02):
        Re = 1.0 / nu
        try:
            info, snaps, ens, spec = run_resolution(
                N=64, nu=nu, dt=0.02, T_warm=60.0, T_collect=400.0,
                snap_every=10, seed=0)
        except Exception as e:
            print(f"  {Re:>5.0f}{nu:>7.3f}   运行失败 {type(e).__name__}: {e}")
            continue
        X = np.asarray(snaps, dtype=np.float64)
        if X.ndim > 2:
            X = X.reshape(len(X), -1)
        if len(X) > 3000:
            X = X[:: max(1, len(X) // 3000)]
        s = np.linalg.svd(X - X.mean(0), compute_uv=False)
        lam = s ** 2 / (s ** 2).sum()
        deff = float(1 / (lam ** 2).sum())
        fluct = float(np.asarray(ens).std() / max(np.asarray(ens).mean(), 1e-30) * 100)
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
        # 检测器 + P3
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
        print(f"  {Re:>5.0f}{nu:>7.3f}{64:>5}{len(X):>7}{deff:>8.2f}{fluct:>11.3f}"
              f"{mg:>12.4f}{str(bfam):>18}{str(de):>6}{s_o:>8.3f}{s_s:>8.3f}  "
              f"{'**A 级候选**' if (P0 and mg > 0 and P3) else ('混沌' if P0 else '非混沌')}"
              f"{'' if mg > 0 else ' + 判据拒'}")
        OUT[f"Re{Re:.0f}"] = dict(Re=Re, nu=nu, d_eff=deff, fluct=fluct,
                                  margin=mg, family=bfam, d=d, sigma_min=sm,
                                  R_max=rm, elbow=de, s_orig=s_o, s_shuf=s_s,
                                  P0=bool(P0), P3=bool(P3))
    json.dump(OUT, open(os.path.join(HERE, "v98_kolmogorov_lowRe.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print("\n  判读：`d_eff` 随 Re 下降是 P0 前提；若某低 Re 处**同时** d_eff>5、margin>0、P3 通过，")
    print("        则 2D Kolmogorov 成为 A 级候选（而且是竞争文献所用的系统）。")
    print("\nsaved v98_kolmogorov_lowRe.json")


if __name__ == "__main__":
    main()
