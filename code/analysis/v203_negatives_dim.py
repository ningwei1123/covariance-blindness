# -*- coding: utf-8 -*-
"""V203：负面系统的 Kaplan--Yorke 维数 —— 判定负面结论是"物理正确"还是"检测失败"。

逻辑
  · D_KY 接近 N（环境维数） -> 吸引子本就高维，无低维图结构，负面结论物理正确
  · D_KY 远小于 N            -> 存在低维吸引子，我们的负面是检测失败（我方的失误）
"""
import os, sys
import numpy as np

H = r"E:\deepseekworkspace\experiments_v12"
sys.path.insert(0, H)


def l96_tangent(N, F=8.0, dt=0.01, n=40000, warm=8000, seed=3):
    """单尺度 Lorenz-96 的 Lyapunov 谱（Benettin QR）。"""
    rng = np.random.default_rng(seed)
    x = F + rng.standard_normal(N)
    x[0] += 1.0

    def f(u):
        return (np.roll(u, -1) - np.roll(u, 2)) * np.roll(u, 1) - u + F

    def J(u):
        M = np.zeros((N, N))
        for i in range(N):
            M[i, (i + 1) % N] += u[(i - 2) % N]
            M[i, (i - 2) % N] -= u[(i + 1) % N]
            M[i, i] -= 1.0
            M[i, (i + 1) % N] += 0.0
        # 由 u_{k-1}(u_{k+1}-u_{k-2}) 的偏导：对 u_{k-1} 是 (u_{k+1}-u_{k-2})，
        # 对 u_{k+1} 是 u_{k-1}，对 u_{k-2} 是 -u_{k-1}
        M = np.zeros((N, N))
        for k in range(N):
            km1, kp1, km2 = (k - 1) % N, (k + 1) % N, (k - 2) % N
            M[k, km1] += u[kp1] - u[km2]
            M[k, kp1] += u[km1]
            M[k, km2] -= u[km1]
            M[k, k] -= 1.0
        return M

    for _ in range(warm):
        k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    Q = np.eye(N)
    s = np.zeros(N)
    for _ in range(n):
        M = J(x)
        k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Q, R = np.linalg.qr(Q + dt * (M @ Q))
        d = np.diag(R)
        s += np.log(np.abs(d) + 1e-300)
    return s / (n * dt)


def dky(lam):
    lam = np.sort(lam)[::-1]
    c = np.cumsum(lam)
    j = int(np.sum(c >= 0))
    if j == 0:
        return 0.0
    if j >= len(lam):
        return float(j)
    return float(j + c[j - 1] / abs(lam[j]))


print("=" * 92)
print("V203  负面系统的 Kaplan--Yorke 维数")
print("=" * 92)
print(f"  {'系统':<18}{'N':>5}{'λ1':>10}{'Σλ':>10}{'正指数':>8}{'D_KY':>9}{'D_KY/N':>9}  判读")
print("  " + "-" * 78)
for N in (10, 20, 40):
    try:
        lam = l96_tangent(N)
    except Exception as e:
        print(f"  L96 N={N:<13} 失败 {type(e).__name__}"); continue
    d = dky(lam)
    pos = int(np.sum(lam > 0))
    ratio = d / N
    verdict = ("**高维，负面物理正确**" if ratio > 0.6 else
               "**低维！负面是检测失败**" if ratio < 0.4 else "临界")
    print(f"  {'L96 N=' + str(N):<18}{N:>5}{lam[0]:>10.4f}{lam.sum():>10.3f}"
          f"{pos:>8}{d:>9.2f}{ratio:>9.2f}  {verdict}")

print()
print("  对标：KS 的 D_KY = 4.22 / 8.95 / 13.80（N=64/64/192）")
print("        D_KY/N = 0.066 / 0.140 / 0.072  <- 远小于 1，故 KS 存在低维结构 ✓")
print(f"        L96 的解析散度应为 -N（Σλ = -N）")
for N in (10, 20, 40):
    print(f"        N={N}: 目标 Σλ = -{N}")
