# -*- coding: utf-8 -*-
"""V88（方向 C 续）：换先验子空间 —— 光滑性先验（Chebyshev）能否救回 DDE？

V87 的诊断
  `V` = 线性化不稳定特征模时，sigma_min 只有 0.008-0.09 ⇒ 与数据 POD 方向严重错位。
  但对时滞方程，状态是 [−tau,0] 上的**光滑函数**，故"低阶多项式"才是自然的先验子空间。

本脚本对同一批 MG 数据比较三种先验子空间
  (a) 线性化不稳定特征模（V87）
  (b) [−tau,0] 上的前 d 个 Chebyshev 多项式
  (c) 前 d 个 Fourier 模（周期化）
并报告 sigma_min / R_max / margin。
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v87_dde_search import integrate_mg, unstable_modes   # noqa: E402
from v50_graph_slope import graph_slope                   # noqa: E402

OUT = {}


def cheb_basis(M, d):
    th = np.linspace(-1.0, 1.0, M)
    cols = [np.ones(M)]
    if d > 1:
        cols.append(th.copy())
    for k in range(2, d):
        cols.append(2 * th * cols[-1] - cols[-2])
    return np.array(cols[:d]).T


def four_basis(M, d):
    th = np.linspace(0, 2 * np.pi, M, endpoint=False)
    cols = [np.ones(M)]
    k = 1
    while len(cols) < d:
        cols.append(np.cos(k * th)); 
        if len(cols) < d:
            cols.append(np.sin(k * th))
        k += 1
    return np.array(cols[:d]).T


def margin_for(X, V, excl):
    Vq = np.linalg.qr(V)[0]
    d = min(Vq.shape[1], X.shape[1] - 1)
    Vq = Vq[:, :d]
    _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    smin = float(np.linalg.svd(Vt[:d] @ Vq, compute_uv=False)[-1])
    gs = graph_slope(X, Vq, excl=excl)
    return smin, gs["R_max"], smin - gs["R_max"], d


def main():
    print("=" * 104)
    print("V88  三种先验子空间下的判据（Mackey-Glass）")
    print("=" * 104)
    M = 64
    for tau in (17.0, 23.0, 30.0, 40.0):
        roots, xs, fp = unstable_modes(tau, M=M)
        X = integrate_mg(tau=tau, M=M)
        if len(X) < 500:
            continue
        X = X - X.mean(0)
        excl = max(5, int(0.05 * tau / 0.01))
        th = np.linspace(-tau, 0.0, M)
        # (a) 不稳定模
        cols = []
        for r in roots:
            v = np.exp(r * th); cols.append(v.real)
            if abs(r.imag) > 1e-8:
                cols.append(v.imag)
        Va = np.array(cols).T if cols else np.zeros((M, 1))
        # (b) Chebyshev，(c) Fourier
        Vb = cheb_basis(M, max(2, Va.shape[1]))
        Vc = four_basis(M, max(2, Va.shape[1]))
        print(f"\n  ── tau = {tau}   dim V = {Va.shape[1]}   样本 {len(X)}")
        print(f"     {'先验子空间':<22}{'sigma_min':>11}{'R_max':>10}{'margin':>11}  判读")
        for name, V in (("线性化不稳定模", Va), ("Chebyshev 多项式", Vb),
                        ("Fourier 模", Vc)):
            try:
                sm, rm, mg, d = margin_for(X, V, excl)
            except Exception as e:
                print(f"     {name:<22}  失败 {type(e).__name__}"); continue
            print(f"     {name:<22}{sm:>11.4f}{rm:>10.4f}{mg:>11.4f}"
                  f"  {'**认证**' if mg > 0 else '拒绝'}")
            OUT[f"tau{tau}_{name}"] = dict(tau=tau, dim_V=int(d), sigma_min=sm,
                                           R_max=rm, margin=mg)
    json.dump(OUT, open(os.path.join(HERE, "v88_dde_priors.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nsaved v88_dde_priors.json")


if __name__ == "__main__":
    main()
