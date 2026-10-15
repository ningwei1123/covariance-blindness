# -*- coding: utf-8 -*-
"""V205：修掉两个 bug 后重跑。

Bug 1  build_V_modes 的复共轭配对错误（写成索引 i 与 i+1）
       -> 64x64 网格 indexing='ij' 展平后，共轭索引是 ((n-i)%n)*n + ((n-j)%n)
       -> 已改为直接构造实的 cos/sin 模

Bug 2  CGLE 混沌判据用了振幅谱（CGLE 的混沌在相位）
       -> 改用最大 Lyapunov 指数（Benettin 双轨重正化）
"""
import os, sys
import numpy as np

H = r"E:\deepseekworkspace\experiments_v12"
sys.path.insert(0, H)
from v50_graph_slope import graph_slope

# ==================================================================== Bug 1
print("=" * 98)
print("实验 1（已修） 2D Kolmogorov：按正确复共轭配对构造波数壳先验基")
print("=" * 98)

X = np.asarray(np.load(os.path.join(H, "v46_ns2d_N64_nu0.02_Y.npy"),
                       mmap_mode="r")[:2000], dtype=np.float64)
N = X.shape[1]
n = int(round(np.sqrt(N)))
print(f"  数据 {X.shape}，网格 {n}x{n}")

kv = np.fft.fftfreq(n, d=1.0 / n)
xi = np.arange(n)[:, None] * np.ones((1, n))          # 行索引 i
eta = np.ones((n, 1)) * np.arange(n)[None, :]         # 列索引 j
KX = kv[xi.astype(int)]
KY = kv[eta.astype(int)]
KABS = np.sqrt(KX ** 2 + KY ** 2).ravel()

# 实的 cos/sin 模：对每个非自共轭的 (i,j) 给两个正交实向量
PAIRS = []
for i in range(n):
    for j in range(n):
        ii, jj = (n - i) % n, (n - j) % n
        if (i, j) < (ii, jj):                          # 每对只取一次
            PAIRS.append((i, j, KABS[i * n + j]))
PAIRS.sort(key=lambda t: t[2])
print(f"  非自共轭波数对总数 {len(PAIRS)}，|k| 范围 {PAIRS[0][2]:.3f}..{PAIRS[-1][2]:.3f}")


def shell_V(lo, hi):
    """|k| 在 [lo,hi) 的实 cos/sin 基。"""
    sel = [(i, j) for (i, j, kk) in PAIRS if lo <= kk < hi]
    cols = []
    th = 2 * np.pi * (xi * 0 + eta * 0)
    for (i, j) in sel:
        ph = 2 * np.pi * (kv[i] * xi + kv[j] * eta) / n
        cols.append(np.cos(ph).ravel())
        cols.append(np.sin(ph).ravel())
    if not cols:
        return None
    return np.column_stack(cols)


def shell_V_low(hi):
    """最低 |k| 的实 cos/sin 基（对照）。"""
    return shell_V(0.0, hi)


def eval_margin(X, V, excl):
    Vq = np.linalg.qr(V)[0]
    d = min(Vq.shape[1], X.shape[1] - 1)
    Vq = Vq[:, :d]
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    smin = float(np.linalg.svd(Vt[:d] @ Vq, compute_uv=False)[-1])
    R = graph_slope(X, Vq, excl=excl)["R_max"]
    return smin, R, smin - R, d


excl = max(20, len(X) // 50)
print(f"  excl={excl}")
print(f"\n  {'先验基':<32}{'d':>5}{'σ_min':>10}{'R_max':>10}{'margin':>11}")
print("  " + "-" * 70)

best = (-1e18, None)
for name, V in [
    ("最低 |k|<2.5（原用）", shell_V_low(2.5)),
    ("最低 |k|<3.5",         shell_V_low(3.5)),
    ("k≈4 壳 [3,5)",         shell_V(3.0, 5.0)),
    ("k≈4 壳 [2.5,5.5)",     shell_V(2.5, 5.5)),
    ("k≈4 壳 [3,7)",         shell_V(3.0, 7.0)),
    ("k≈8 壳 [7,10)",        shell_V(7.0, 10.0)),
    ("全谱 |k|<12",          shell_V(0.0, 12.0)),
]:
    if V is None:
        print(f"  {name:<32}  （无模式）"); continue
    try:
        sm, R, mg, d = eval_margin(X, V, excl)
    except Exception as e:
        print(f"  {name:<32}  失败 {type(e).__name__}"); continue
    flag = "  <-- 转正!" if mg > 0 else ""
    print(f"  {name:<32}{d:>5}{sm:>10.4f}{R:>10.3f}{mg:>+11.4f}{flag}")
    if mg > best[0]:
        best = (mg, name)
print(f"\n  最佳 margin = {best[0]:+.4f}  ({best[1]})")

# ==================================================================== Bug 2
print()
print("=" * 98)
print("实验 2（已修） CGLE 的最大 Lyapunov 指数（Benettin 双轨重正化）")
print("=" * 98)


def etdrk4_coeffs(lam, dt, M=32):
    """Kassam--Trefethen 围道积分的 ETDRK4 系数。"""
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


def cgle_step_factory(b, c, L, N, dt):
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


def largest_lyap(b, c, L=200.0, N=512, dt=0.02, T_warm=200.0, T_meas=400.0,
                 renorm=10, seed=3, amp=0.1):
    rng = np.random.default_rng(seed)
    step = cgle_step_factory(b, c, L, N, dt)
    A = np.fft.fft(amp * (rng.standard_normal(N) + 1j * rng.standard_normal(N)))
    for _ in range(int(T_warm / dt)):
        A = step(A)
        if not np.all(np.isfinite(A)):
            return None
    # 切向量：有限差分
    dA = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) * 1e-8
    A2 = A + dA
    s = 0.0
    nst = int(T_meas / dt)
    for it in range(nst):
        A = step(A); A2 = step(A2)
        if it % renorm == renorm - 1:
            diff = A2 - A
            nrm = np.linalg.norm(diff)
            if nrm == 0 or not np.isfinite(nrm):
                return None
            s += np.log(nrm)
            A2 = A + diff / nrm * 1e-8
    return s / (nst * dt)


for (b, c, L) in ((2.0, 2.0, 200.0), (0.5, 5.0, 300.0), (2.0, -2.0, 200.0)):
    lam = largest_lyap(b, c, L)
    if lam is None:
        print(f"  b={b}, c={c}: 发散/失败")
        continue
    verdict = ("**λ1 > 0 => 混沌**" if lam > 1e-3
               else "**λ1 ≈ 0 => 非混沌（周期/准周期）**" if lam > -1e-3
               else "**λ1 < 0 => 稳定不动点/极限环**")
    print(f"  b={b:<4} c={c:<5} L={L:<6.0f}  λ1 = {lam:+.6f}   {verdict}")
