# -*- coding: utf-8 -*-
"""V95：负面结论的**多先验复检**。

动机（由 Mackey-Glass 暴露）
  判据 margin = sigma_min(P_{W_d}|V) - R_max 的结论**强烈依赖先验子空间 V**：
  MG 用线性化不稳定模全拒（sigma_min 0.008-0.09），用 Chebyshev 光滑先验则三个认证。
  故凡是**只用过一种 V** 的负面结论，都可能只是先验选错的产物。

被复检的负面
  · single-scale Lorenz-96（N=20, 40）
  · Swift-Hohenberg（r=1.5,c=16 与 r=2.0,c=24）
  · 2D Kolmogorov flow（N=64, nu=0.02）
  · 1D complex Ginzburg-Landau

五个先验族（**都由物理/光滑性定义，不含数据信息**）
  P1 low-fourier    低波数 Fourier（原先唯一用过的）
  P2 high-fourier   高波数 Fourier（L96 的能量在小尺度，低波数很可能就是错的先验）
  P3 chebyshev      指标轴上的低阶 Chebyshev（光滑性先验，MG 的成功先验）
  P4 random         随机子空间（**对照**：应当被拒，用以校准判据不致过于宽松）
  P5 growing-fourier 几何增长波数（偏向小尺度的折中）

对每个 (系统, 先验族) 扫描 d，报告 margin 的最大值与取到的 d。
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v50_graph_slope import graph_slope     # noqa: E402

OUT = {}


# ---------------------------------------------------------------- 先验子空间族
def low_fourier(N, d, L=2 * np.pi):
    x = np.arange(N) * (L / N)
    cols, k = [], 1
    while len(cols) < d:
        cols.append(np.cos(2 * np.pi * k * x / L))
        if len(cols) < d:
            cols.append(np.sin(2 * np.pi * k * x / L))
        k += 1
    return np.array(cols[:d]).T


def high_fourier(N, d, L=2 * np.pi):
    """取**最高**的 d/2 个波数对。"""
    x = np.arange(N) * (L / N)
    kmax = N // 2 - 1
    cols, k = [], kmax
    while len(cols) < d and k > 0:
        cols.append(np.cos(2 * np.pi * k * x / L))
        if len(cols) < d:
            cols.append(np.sin(2 * np.pi * k * x / L))
        k -= 1
    return np.array(cols[:d]).T


def chebyshev(N, d):
    t = np.linspace(-1.0, 1.0, N)
    cols = [np.ones(N)]
    if d > 1:
        cols.append(t.copy())
    for k in range(2, d):
        cols.append(2 * t * cols[-1] - cols[-2])
    return np.array(cols[:d]).T


def random_sub(N, d, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, d))


def growing_fourier(N, d, L=2 * np.pi):
    """波数按几何级数增长：偏向小尺度但不极端。"""
    x = np.arange(N) * (L / N)
    ks = np.unique(np.round(np.geomspace(1, max(2, N // 4), max(1, d // 2))).astype(int))
    cols = []
    for k in ks:
        cols.append(np.cos(2 * np.pi * k * x / L))
        if len(cols) < d:
            cols.append(np.sin(2 * np.pi * k * x / L))
    while len(cols) < d:
        cols.append(np.cos(2 * np.pi * (ks[-1] + len(cols)) * x / L))
    return np.array(cols[:d]).T


FAMILIES = {
    "low-fourier": low_fourier,
    "high-fourier": high_fourier,
    "chebyshev": chebyshev,
    "random(control)": random_sub,
    "growing-fourier": growing_fourier,
}


def margin_scan(X, fam, ds, excl):
    """对给定先验族扫描 d，返回 (最佳 margin, 对应 d, 该处 sigma_min, R_max)。"""
    _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    best = (-1e30, None, None, None)
    for d in ds:
        if d >= min(Vt.shape[0], X.shape[1]):
            continue
        try:
            V = FAMILIES[fam](X.shape[1], d)
        except Exception:
            continue
        Vq = np.linalg.qr(V)[0][:, :d]
        smin = float(np.linalg.svd(Vt[:d] @ Vq, compute_uv=False)[-1])
        gs = graph_slope(X, Vq, excl=excl)
        mg = smin - gs["R_max"]
        if mg > best[0]:
            best = (mg, d, smin, gs["R_max"])
    return best


def load(name, cap=4000):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return None
    X = np.load(p, mmap_mode="r")
    X = np.asarray(X, dtype=np.float64)
    if X.ndim > 2:
        X = X.reshape(len(X), -1)
    if len(X) > cap:
        X = X[:: max(1, len(X) // cap)]
    return X


def main():
    print("=" * 112)
    print("V95  负面结论的多先验复检（五个先验族，扫描 d 取最佳 margin）")
    print("=" * 112)
    SYSTEMS = [
        ("single-scale L96 N=20", "v21_snap_L96_N20.npy", 400),
        ("single-scale L96 N=40", "v21_snap_L96_N40.npy", 400),
        ("Swift-Hohenberg r=1.5 c=16", "v25_snap_SH_r1.5_c16.npy", 50),
        ("Swift-Hohenberg r=2.0 c=24", "v25_snap_SH_r2.0_c24.npy", 50),
        ("2D Kolmogorov N=64 nu=0.02", "v46_ns2d_N64_nu0.02_Y.npy", 30),
    ]
    for label, fn, excl in SYSTEMS:
        X = load(fn)
        if X is None:
            print(f"\n  [缺] {label} ({fn})"); continue
        s = np.linalg.svd(X - X.mean(0), compute_uv=False)
        lam = s ** 2 / (s ** 2).sum()
        deff = float(1 / (lam ** 2).sum())
        print(f"\n  ── {label}   形状 {X.shape}   d_eff = {deff:.2f}")
        print(f"     {'先验族':<18}{'最佳 margin':>13}{'于 d=':>7}{'sigma_min':>11}"
              f"{'R_max':>10}  判读")
        ds = list(range(2, min(30, X.shape[1] - 1)))
        for fam in FAMILIES:
            mg, d, sm, rm = margin_scan(X, fam, ds, excl)
            print(f"     {fam:<18}{mg:>13.4f}{str(d):>7}{sm:>11.4f}{rm:>10.4f}"
                  f"  {'**认证**' if mg > 0 else '拒绝'}")
            OUT[f"{label}|{fam}"] = dict(margin=mg, d=d, sigma_min=sm, R_max=rm,
                                         d_eff=deff, shape=list(X.shape))
    json.dump(OUT, open(os.path.join(HERE, "v95_multi_prior.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print("\n  判读要点：")
    print("    · `random(control)` 若不给出正 margin，说明判据没有被宽松到失去鉴别力；")
    print("    · 若某个**物理先验**给出显著正 margin，则该负面须重新审视（可能只是先验选错）。")
    print("\nsaved v95_multi_prior.json")


if __name__ == "__main__":
    main()
