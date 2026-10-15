# -*- coding: utf-8 -*-
"""V204：两个收尾实验
  实验1  2D Kolmogorov：把先验基从「最低波数」改为「强迫尺度 k≈4 的波数壳」
  实验2  CGLE (b=2.0, c=2.0)：确认是否混沌
"""
import os, sys, time
import numpy as np

H = r"E:\deepseekworkspace\experiments_v12"
sys.path.insert(0, H)
from v50_graph_slope import graph_slope

# ============================================================ 实验 1
print("=" * 96)
print("实验 1  2D Kolmogorov：先验基的波数尺度")
print("=" * 96)

X = np.asarray(np.load(os.path.join(H, "v46_ns2d_N64_nu0.02_Y.npy"),
                       mmap_mode="r")[:2000], dtype=np.float64)
N = X.shape[1]
n2 = int(round(np.sqrt(N)))
print(f"  数据 {X.shape}，网格 {n2}x{n2}")

kx = np.fft.fftfreq(n2, d=1.0 / n2)          # 0..31, -32..-1
KX, KY = np.meshgrid(kx, kx, indexing="ij")
KK = np.sqrt(KX ** 2 + KY ** 2).ravel()


def shell_basis(lo, hi):
    """取 |k| 在 [lo, hi) 的 Fourier 模式（实部/虚部成对）。"""
    idx = np.where((KK >= lo) & (KK < hi))[0]
    return idx


def build_V_modes(idxs):
    V = np.zeros((N, 2 * len(idxs)))
    for j, i in enumerate(idxs):
        V[i, 2 * j] = 1.0
        ii = i + 1 if i % 2 == 0 else i - 1
        if ii < N:
            V[ii, 2 * j + 1] = 1.0
    return V


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
print(f"\n  {'先验基':<34}{'d':>4}{'σ_min':>10}{'R_max':>10}{'margin':>11}")
print("  " + "-" * 70)

CASES = [
    ("最低 8 个波数（原用）", shell_basis(0, 2.5)),
    ("最低 16 个波数", shell_basis(0, 3.0)),
    (f"k≈4 壳 [3,5)（强迫尺度）", shell_basis(3.0, 5.0)),
    (f"k≈4 壳 [2,6)", shell_basis(2.0, 6.0)),
    (f"k≈4 壳 [3,6)", shell_basis(3.0, 6.0)),
    (f"k≈4 壳 [3,8)", shell_basis(3.0, 8.0)),
    (f"k≈8 壳 [7,10)", shell_basis(7.0, 10.0)),
]
best_all = (-1e18, None)
for name, idx in CASES:
    if len(idx) == 0:
        print(f"  {name:<34}  （无模式）"); continue
    V = build_V_modes(idx)
    if V.shape[1] < 2:
        continue
    try:
        sm, R, mg, d = eval_margin(X, V, excl)
    except Exception as e:
        print(f"  {name:<34}  失败 {type(e).__name__}"); continue
    flag = "  <-- 转正!" if mg > 0 else ""
    print(f"  {name:<34}{d:>4}{sm:>10.4f}{R:>10.3f}{mg:>+11.4f}{flag}")
    if mg > best_all[0]:
        best_all = (mg, name)
print(f"\n  最佳 margin = {best_all[0]:+.4f}  ({best_all[1]})")

# ============================================================ 实验 2
print()
print("=" * 96)
print("实验 2  CGLE (b=2.0, c=2.0) 是否混沌")
print("=" * 96)
try:
    from v54b_screen_families import cgle_run
except Exception as e:
    print(f"  无法导入 cgle_run: {e}")
    cgle_run = None

if cgle_run is not None:
    for (b, c, L) in ((2.0, 2.0, 200.0), (0.5, 5.0, 300.0)):
        t0 = time.time()
        try:
            Xc = np.asarray(cgle_run(b=b, c=c, L=L, N=512, dt=0.02, T=400.0,
                                     warm=200.0, stride=4, seed=3), dtype=np.float64)
        except Exception as e:
            print(f"  b={b} c={c}: 运行失败 {type(e).__name__}"); continue
        if len(Xc) > 3000:
            Xc = Xc[:: max(1, len(Xc) // 3000)]
        s = np.linalg.svd(Xc - Xc.mean(0), compute_uv=False)
        lam = s ** 2 / (s ** 2).sum()
        deff = float(1 / (lam ** 2).sum())
        # 时间序列的混沌指标：能量谱的宽带性 + 相空间的自相关衰减
        amp = np.abs(Xc).mean(1)
        ac = np.corrcoef(amp[:-1], amp[1:])[0, 1]
        # 宽带性：功率谱中显著峰的占比
        P = np.abs(np.fft.rfft(amp - amp.mean())) ** 2
        P = P[1:]
        peaks = int(np.sum(P > 0.1 * P.max()))
        frac = peaks / len(P)
        print(f"\n  b={b}, c={c}, L={L:.0f}   n={len(Xc)}  ({time.time()-t0:.0f}s)")
        print(f"    d_eff = {deff:.3f}")
        print(f"    自相关(lag=1) = {ac:.4f}")
        print(f"    谱峰占比 = {frac*100:.1f}%  (>20% 通常是宽带混沌)")
        verdict = ("**宽带谱 + 弱自相关 => 混沌**" if frac > 0.2 and ac < 0.9
                   else "**窄带/强自相关 => 疑非混沌（周期或准周期）**")
        print(f"    判读: {verdict}")
