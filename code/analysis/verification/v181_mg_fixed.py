# -*- coding: utf-8 -*-
"""V181：修正 DDE 积分器并重做 Mackey–Glass 独立复算。

V179/V180 的错误
  `x_at` 从 `buf` 取值，而 `buf` 在首步为空 ⇒ 退化为返回不动点 FP，
  于是 x ≡ 1.0 恒定，std = 0，谱全零，margin 全 0。

本版
  用「整数步索引 + 完整历史数组」实现标准的步进法 RK4：
    traj[i] 表示 t = i·dt 的状态，i 从 −n_hist 一直到当前；
    x(t−τ) 取 traj[i − round(τ/dt)]。
"""
import json, os
import numpy as np

H = r"E:\deepseekworkspace\experiments_v12"
OUT = {}

BETA, GAMMA, NN = 0.2, 0.1, 10.0
FP = (BETA / GAMMA - 1.0) ** (1.0 / NN)
FPRIME = BETA * (1 + FP ** NN - NN * FP ** NN) / (1 + FP ** NN) ** 2


def char_roots(tau, nroot=14):
    roots = []
    for k in range(nroot):
        lam = complex(-GAMMA, (2 * k + 1) * np.pi / (2 * tau))
        for _ in range(300):
            e = np.exp(-lam * tau)
            f = lam + GAMMA - FPRIME * e
            df = 1 + FPRIME * tau * e
            step = f / df
            lam -= step
            if abs(step) < 1e-15:
                break
        roots.append(lam)
    return roots


def mg_run(tau, M=64, dt=0.01, T=400.0, warm=250.0, seed=3):
    rng = np.random.default_rng(seed)
    ntau = int(round(tau / dt))
    nhist = ntau + 2
    # 初始历史：围绕不动点的平滑扰动（用几个 Fourier 模，避免过度粗糙）
    s = np.arange(nhist) / nhist
    hist = FP + 0.05 * sum(
        a * np.sin(2 * np.pi * (k + 1) * s + p)
        for k, (a, p) in enumerate(zip(rng.normal(0, 1, 6), rng.random(6) * 2 * np.pi)))
    # xs[j] 对应 t = (j - nhist)*dt；需含 t=0 那一点，故多一个元素
    xs = list(hist) + [hist[-1]]            # 索引 0..nhist，j=nhist 即 t=0
    nstep = int((warm + T) / dt)

    def f(xv, xdv):
        return BETA * xdv / (1.0 + xdv ** NN) - GAMMA * xv

    snaps = []
    snap_every = max(1, int(0.5 / dt))
    for step in range(nstep):
        j = nhist + step                    # 当前时间 t = step*dt 对应 xs[j]
        x = xs[j]
        xd = xs[j - ntau]
        k1 = f(x, xd)
        k2 = f(x + dt / 2 * k1, xd)
        k3 = f(x + dt / 2 * k2, xd)
        k4 = f(x + dt * k3, xd)
        xs.append(x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        if step * dt >= warm and step % snap_every == 0:
            # 历史段 [t−τ, t]
            seg = np.array([xs[j + 1 - ntau + int(round(g / dt))]
                            for g in np.linspace(0, tau, M)])
            snaps.append(seg)
    return np.asarray(snaps, dtype=float)


def v_cheb(M, d):
    s = np.linspace(-1, 1, M)
    cols = [np.ones(M)]
    if d > 1:
        cols.append(s.copy())
    for k in range(2, d):
        cols.append(2 * s * cols[-1] - cols[-2])
    return np.array(cols[:d]).T


def v_fourier(M, d):
    s = np.linspace(0, 2 * np.pi, M, endpoint=False)
    cols, k = [], 1
    while len(cols) < d:
        cols.append(np.cos(k * s))
        if len(cols) < d:
            cols.append(np.sin(k * s))
        k += 1
    return np.array(cols[:d]).T


def v_unstable(M, d, tau):
    roots = [r for r in char_roots(tau) if r.real > 0]
    roots.sort(key=lambda r: -r.real)
    s = np.linspace(-tau, 0.0, M)
    cols = []
    for r in roots:
        cols.append(np.exp(r.real * s) * np.cos(r.imag * s))
        if len(cols) < d:
            cols.append(np.exp(r.real * s) * np.sin(r.imag * s))
    while len(cols) < d:
        cols.append(np.cos((len(cols) + 1) * np.pi * s / tau))
    return np.array(cols[:d]).T


def graph_slope_R(X, V, excl=20, nq=250, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.choice(n, size=min(nq, n), replace=False)
    Vq = np.linalg.qr(V)[0]
    Z = X @ Vq
    worst = 0.0
    for i in idx:
        d2 = ((Z - Z[i]) ** 2).sum(1)
        d2[max(0, i - excl):i + excl + 1] = np.inf
        j = int(np.argmin(d2))
        dV = X[j] - X[i]
        a = Vq.T @ dV
        perp = dV - Vq @ a
        nv, np_ = np.linalg.norm(a), np.linalg.norm(perp)
        if nv > 1e-12:
            worst = max(worst, np_ / nv)
    return float(worst)


def main():
    print("=" * 100)
    print("V181  Mackey–Glass 独立复算（修正积分器）")
    print("=" * 100)
    print(f"  不动点 x* = {FP:.4f}   f'(x*) = {FPRIME:.4f}")
    print(f"\n  {'τ':>5}{'Re λ1':>9}{'d_eff':>8}{'σ_min(Cheb)':>13}{'R':>9}"
          f"{'margin':>10}{'margin(Four)':>14}{'margin(Unst)':>14}")
    for tau in (17.0, 23.0, 30.0, 40.0):
        l1 = max(r.real for r in char_roots(tau))
        X = mg_run(tau)
        if len(X) < 30 or X.std() < 1e-8:
            print(f"  {tau:>5}{l1:>9.4f}   数据退化 (n={len(X)}, std={X.std():.2e})")
            continue
        s = np.linalg.svd(X - X.mean(0), compute_uv=False)
        lam = s ** 2 / (s ** 2).sum()
        deff = float(1 / (lam ** 2).sum())
        _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
        res = {}
        for nm, V in (("cheb", v_cheb(64, 8)), ("four", v_fourier(64, 8)),
                      ("unst", v_unstable(64, 8, tau))):
            Vq = np.linalg.qr(V)[0][:, :8]
            smin = float(np.linalg.svd(Vt[:8] @ Vq, compute_uv=False)[-1])
            R = graph_slope_R(X, Vq)
            res[nm] = dict(sigma_min=smin, R=R, margin=smin - R)
        c, fr, u = res["cheb"], res["four"], res["unst"]
        print(f"  {tau:>5}{l1:>9.4f}{deff:>8.2f}{c['sigma_min']:>13.4f}{c['R']:>9.4f}"
              f"{c['margin']:>+10.4f}{fr['margin']:>+14.4f}{u['margin']:>+14.4f}")
        OUT[f"tau={tau}"] = dict(lam1=l1, d_eff=deff, n=len(X), **res)

    json.dump(OUT, open(os.path.join(H, "v181_mg_indep.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print("\n  验收：数据应为混沌（d_eff > 5）；Chebyshev 先验下 τ=23 应为唯一显著正值。")
    print("saved v181_mg_indep.json")


if __name__ == "__main__":
    main()
