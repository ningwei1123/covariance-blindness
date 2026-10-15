# -*- coding: utf-8 -*-
"""V206：闭合 CGLE —— 用文献判据（相位滑移）+ 标度收敛的 Lyapunov 指数。

判据来源：Riecke 讲义 —— CGLE 的振幅湍流表现为「相位滑移，|A| 趋于 0 的事件
不规则且持续出现」。故检验对象是 |A| 的极小值事件，不是振幅谱。

Bug 2 的真正病因：有限差分扰动的**标度**。A 的傅里叶系数幅度 ~ N*amp = 51，
而扰动仅 1e-8（相对 2e-10），经 ETDRK4 多步加减后被舍入淹没 -> λ1 = -92 是噪声。
修法：扰动按 ||A|| 的相对比例设定，并做**扰动幅度收敛性检验**。
"""
import os, sys
import numpy as np

H = r"E:\deepseekworkspace\experiments_v12"
sys.path.insert(0, H)


def etdrk4_coeffs(lam, dt, M=32):
    E = np.exp(dt * lam)
    E2 = np.exp(dt * lam / 2)
    LR = lam.reshape(-1, 1)
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LRr = dt * LR + r.reshape(1, -1)
    Q = dt * np.real(np.mean((np.exp(LRr / 2) - 1) / LRr, axis=1))
    f1 = dt * np.real(np.mean((-4 - LRr + np.exp(LRr) * (4 - 3 * LRr + LRr ** 2))
                              / LRr ** 3, axis=1))
    f2 = dt * np.real(np.mean((2 + LRr + np.exp(LRr) * (-2 + LRr)) / LRr ** 3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * LRr - LRr ** 2 + np.exp(LRr) * (4 - LRr))
                              / LRr ** 3, axis=1))
    return E, E2, Q, f1, f2, f3


def make_step(b, c, L, N, dt):
    k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    lam = 1.0 - (1.0 + 1j * b) * k ** 2
    E, E2, Q, f1, f2, f3 = etdrk4_coeffs(lam, dt)

    def Nl(Ah):
        a = np.fft.ifft(Ah)
        return -np.fft.fft((1.0 + 1j * c) * np.abs(a) ** 2 * a)

    def step(A):
        N1 = Nl(A); a = E2 * A + Q * N1; N2 = Nl(a)
        bb = E2 * A + Q * N2; N3 = Nl(bb)
        cc = E2 * a + Q * (2 * N3 - N1); N4 = Nl(cc)
        return E * A + f1 * N1 + 2 * f2 * (N2 + N3) + f3 * N4

    return step


def run(b, c, L, N=512, dt=0.02, T_warm=200.0, T_meas=400.0, eps_rel=1e-6,
        renorm=1, seed=3, amp=0.1):
    rng = np.random.default_rng(seed)
    step = make_step(b, c, L, N, dt)
    A = np.fft.fft(amp * (rng.standard_normal(N) + 1j * rng.standard_normal(N)))
    for _ in range(int(T_warm / dt)):
        A = step(A)
        if not np.all(np.isfinite(A)):
            return None, None, None
    # 相位滑移统计（在测量窗口内）
    slips = 0
    mins = []
    for it in range(int(T_meas / dt)):
        A = step(A)
        a = np.abs(np.fft.ifft(A))
        mn = float(a.min())
        mins.append(mn)
        if mn < 0.35 * float(np.sqrt((a ** 2).mean())):
            slips += 1
    # Lyapunov：相对标度的扰动 + 每步重正化
    nrmA = np.linalg.norm(A)
    rng2 = np.random.default_rng(seed + 1)
    dA = (rng2.standard_normal(N) + 1j * rng2.standard_normal(N))
    dA = dA / np.linalg.norm(dA) * eps_rel * nrmA
    A2 = A + dA
    s = 0.0
    nst = int(T_meas / dt)
    for it in range(nst):
        A = step(A); A2 = step(A2)
        if it % renorm == renorm - 1:
            diff = A2 - A
            nr = np.linalg.norm(diff)
            if nr == 0 or not np.isfinite(nr):
                return None, slips, mins
            s += np.log(nr / (eps_rel * np.linalg.norm(A)))
            A2 = A + diff / nr * eps_rel * np.linalg.norm(A)
    return s / (nst * dt), slips, mins


print("=" * 100)
print("V206  闭合 CGLE：相位滑移 + 标度收敛的 Lyapunov 指数")
print("=" * 100)
print("  文献判据：振幅湍流 = 相位滑移（|A|->0）不规则且持续地出现\n")

CASES = [(2.0, 2.0, 200.0), (0.5, 5.0, 300.0), (2.0, -2.0, 200.0)]
for (b, c, L) in CASES:
    bc = b * c
    print(f"  ── b={b}, c={c}, L={L:.0f}   (bc={bc:+.1f}, 1+bc={1+bc:+.1f})")
    # 扰动幅度收敛性检验
    print(f"     {'eps_rel':>10}{'λ1':>14}   判读")
    lams = []
    slips, mins = None, None
    for eps in (1e-7, 1e-6, 1e-5):
        lam, sl, mn = run(b, c, L, eps_rel=eps)
        if lam is None:
            print(f"     {eps:>10.0e}   发散"); continue
        lams.append(lam); slips, mins = sl, mn
        print(f"     {eps:>10.0e}{lam:>+14.6f}")
    if lams:
        spread = max(lams) - min(lams)
        conv = "**收敛**" if spread < 0.02 * (abs(np.mean(lams)) + 1e-6) else "**未收敛**"
        lam = float(np.mean(lams))
        print(f"     -> λ1 = {lam:+.6f}  (跨 eps 的散布 {spread:.2e}, {conv})")
        if mins is not None:
            print(f"     |A| 最小值 = {min(mins):.5f}，滑移计数 = {slips}")
        v = ("**混沌**（λ1>0 且有相位滑移）" if lam > 1e-3 and slips and slips > 5
             else "**非混沌**（λ1<=0）" if lam <= 1e-3
             else "**疑混沌**（λ1>0 但相位滑移少）")
        print(f"     判读: {v}\n")

print("=" * 100)
print("  文献预期对照（BF 条件 1+bc<0 才不稳定）")
print("    b=2.0, c=2.0   bc=+4.0 -> BF 稳定 -> 预期非混沌")
print("    b=0.5, c=5.0   bc=+2.5 -> BF 稳定 -> 预期非混沌（b 不够大）")
print("    b=2.0, c=-2.0  bc=-4.0 -> BF 不稳定 -> 预期混沌")
