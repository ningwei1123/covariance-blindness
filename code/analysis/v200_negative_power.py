# -*- coding: utf-8 -*-
"""V200：负面结论的统计功效检验（power test）。

对每个负面系统：保留头部，把尾部替换为**已知维数 d0 的合成图**，
再跑同一条判据与估计量，看流程能否给出正例。
"""
import os, sys
import numpy as np

H = r"E:\deepseekworkspace\experiments_v12"
sys.path.insert(0, H)
import v20_robust
from v50_graph_slope import graph_slope
from v100_ten_checks import (idx_fourier_low, idx_fourier_high, idx_cheb,
                             idx_fourier_mid, idx_rand)

NEG = {
    "single-scale L96 N=20": ("v21_snap_L96_N20.npy", "low-fourier"),
    "single-scale L96 N=40": ("v21_snap_L96_N40.npy", "low-fourier"),
    "2D Kolmogorov N=64":    ("v46_ns2d_N64_nu0.02_Y.npy", "low-fourier"),
    "Swift-Hohenberg r=1.5": ("v25_snap_SH_r1.5_c16.npy", "low-fourier"),
}
BASES = {"low-fourier": idx_fourier_low, "high-fourier": idx_fourier_high,
         "chebyshev": idx_cheb, "mid-fourier": idx_fourier_mid, "random": idx_rand}


def build_V(N, idx):
    V = np.zeros((N, len(idx)))
    for j, i in enumerate(idx):
        V[j, j] = 1.0
    return V


def inject(X, d0, amp=0.30, seed=0):
    rng = np.random.default_rng(seed)
    Xc = X - X.mean(0)
    Vt = np.linalg.svd(Xc, full_matrices=False)[2]
    z = Xc @ Vt[:d0].T
    z = (z - z.mean(0)) / (z.std(0) + 1e-12)
    N = X.shape[1]
    W = rng.standard_normal((N, d0))
    W = W - Vt[:d0].T @ (Vt[:d0] @ W)
    g = np.tanh(z) @ W.T
    g = g / (np.linalg.norm(g) + 1e-30) * amp * np.linalg.norm(z @ Vt[:d0])
    return Xc + g


def margin(X, d, excl, kind):
    idx = BASES[kind](X.shape[1], d)
    Vq = np.linalg.qr(build_V(X.shape[1], idx))[0][:, :d]
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    smin = float(np.linalg.svd(Vt[:d] @ Vq, compute_uv=False)[-1])
    return smin - graph_slope(X, Vq, excl=excl)["R_max"]


print("=" * 100)
print("V200  负面结论的统计功效检验（原数据 vs 注入已知维数的合成图）")
print("=" * 100)

for name, (f, kind0) in NEG.items():
    p = os.path.join(H, f)
    if not os.path.exists(p):
        print(f"\n  {name}: 缺 {f}"); continue
    X0 = np.asarray(np.load(p, mmap_mode="r")[:2500], dtype=np.float64)
    N = X0.shape[1]
    excl = max(20, len(X0) // 50)
    ds = list(range(2, min(18, N - 1)))

    print(f"\n  ── {name}   N={N}  n={len(X0)}  excl={excl}")
    print(f"     {'条件':<14}{'最佳 margin':>13}{'取正的族':>12}{'肘点':>8}")
    print("     " + "-" * 48)
    for d0 in (None, 3, 5, 8, 12):
        if d0 is not None and d0 >= N - 1:
            continue
        X = X0 if d0 is None else inject(X0, d0)
        best, bk = -1e18, "-"
        for k in BASES:
            try:
                m = margin(X, 8, excl, k)
            except Exception:
                continue
            if m > best:
                best, bk = m, k
        # 全维扫描
        allm = []
        for d in ds:
            try:
                allm.append(margin(X, d, excl, kind0))
            except Exception:
                pass
        if allm and max(allm) > best:
            best = max(allm); bk = kind0
        try:
            _, el = v20_robust.elbow(X, ds, deg=2, nval=min(600, len(X) // 4))
            eb = el[0]
        except Exception:
            eb = None
        tag = "原数据" if d0 is None else f"注入 d0={d0}"
        print(f"     {tag:<14}{best:>+13.4f}{bk:>12}{str(eb):>8}")
    print()

print("  判读：")
print("    · 注入后 margin 转正且肘点≈d0  -> 流程有功效，原负面结论可信")
print("    · 注入任意 d0 后仍为负          -> 流程失灵，原负面结论不可信")
