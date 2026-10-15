# -*- coding: utf-8 -*-
"""V83b（方向 A，修正版）：协方差盲性的普遍定理。

定理
  对任意正定 `C ∈ R^{N×N}` 与任意 `0 < d < N`，存在两个概率测度，协方差**都恰为 `C`**，
  而一个的图维数 `d*(μ_G) = N`，另一个 `d*(μ_M) - d`。
  - `d*` **不是协方差算子的任何函数**。

关键实现要求（V83 的两个 bug 所在）
  1. 在**特征基**里构造，最后必须旋回坐标帧：`X = Z Vᵀ`；
  2. 两个测度都要做**精确重着色** `X ← G C_G^{-1/2} C^{1/2}`，使**样本**协方差恰为 `C`
     （否则比的是两个不同的采样协方差，不是同一个 `C`）。
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}


def exact_recolour(X, C):
    """把样本 X 精确重着色到协方差 C：X ← Xc C_X^{-1/2} C^{1/2}（再补回均值 0）。"""
    Xc = X - X.mean(0)
    Cx = Xc.T @ Xc / len(Xc)
    wx, Vx = np.linalg.eigh(Cx)
    wx = np.clip(wx, 1e-14, None)
    Cxh = (Vx * (1.0 / np.sqrt(wx))[None, :]) @ Vx.T   # C_X^{-1/2}（对称）
    wc, Vc = np.linalg.eigh(C)
    wc = np.clip(wc, 0, None)
    Ch = (Vc * np.sqrt(wc)[None, :]) @ Vc.T            # C^{1/2}（对称）
    return Xc @ (Cxh @ Ch)


def make_gaussian(C, n, seed):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((n, C.shape[0]))
    return exact_recolour(G, C)                    # 样本协方差**恰为** C


def make_graph(C, d, n, seed):
    """u = Φ z + h(z)，Cov(u) 恰为 C，且尾部是头部的函数。"""
    N = C.shape[0]
    w, V = np.linalg.eigh(C)                       # 升序
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]                       # 降序
    rng = np.random.default_rng(seed)
    # 头部：z 的样本协方差恰为 diag(w[:d])
    Rh = rng.standard_normal((n, d))
    zh = exact_recolour(Rh, np.diag(w[:d]))
    # 尾部：每个坐标是一个**二次函数**（故二次图拟合应能完全解释），
    #        先造出与 z 不相关的函数，再精确重着色到 diag(w[d:])
    # 用**互不相同**的单项式（否则 A 秩亏，无法张满尾部）
    import itertools as _it
    from math import comb as _cb
    def _nmono(dd, p):
        return sum(_cb(dd + j - 1, j) for j in range(2, p + 1))
    pdeg = 2
    while _nmono(d, pdeg) < (N - d):
        pdeg += 1
    mono = []
    for deg in range(2, pdeg + 1):   # 从 2 阶起：1 阶项会被线性投影清零
        for al in _it.combinations_with_replacement(range(d), deg):
            mono.append(al)
            if len(mono) >= (N - d):
                break
        if len(mono) >= (N - d):
            break
    A = np.zeros((n, N - d))
    for j, al in enumerate(mono):
        v = np.ones(n)
        for a in al:
            v = v * zh[:, a]
        v = v - v.mean()
        A[:, j] = v / max(v.std(), 1e-300)
    used_pdeg = pdeg
    # 去掉 A 与 z 的线性相关（只去线性部分，保留二次函数结构）
    Zc = zh - zh.mean(0)
    Pz = Zc @ np.linalg.pinv(Zc.T @ Zc) @ Zc.T
    A = A - Pz @ A
    At = exact_recolour(A, np.diag(w[d:]))         # 尾部协方差恰为 diag(w[d:])
    # 旋回坐标帧
    X = zh @ V[:, :d].T + At @ V[:, d:].T
    return X, used_pdeg


def cov_of(X):
    Xc = X - X.mean(0)
    return Xc.T @ Xc / len(Xc)


def deff(X):
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    lam = s ** 2 / (s ** 2).sum()
    return float(1 / (lam ** 2).sum())


def graph_resid(X, d, deg=2):
    """在头 d 个 POD 坐标上做 deg 次图拟合，返回尾部的相对残差。"""
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:d].T
    R = Xc - Z @ Vt[:d]
    F = [Z]
    if deg >= 2:
        for i in range(d):
            for j in range(i, d):
                F.append((Z[:, i] * Z[:, j])[:, None])
    F = np.hstack(F)
    coef, *_ = np.linalg.lstsq(F, R, rcond=None)
    res = R - F @ coef
    return float((res ** 2).sum() / (R ** 2).sum())


def main():
    print("=" * 104)
    print("V83b（方向 A）  协方差盲性的普遍定理：任意 C、任意 d")
    print("=" * 104)
    print(f"  {'N':>4}{'d':>4}{'谱':>10}{'||C_G-C_M||/||C||':>20}{'||C_G-C||':>12}"
          f"{'d_eff(G)':>10}{'d_eff(M)':>10}{'残差G':>8}{'残差M':>8}")
    for N, d, spec in ((14, 4, "lin"), (24, 6, "lin"), (40, 8, "lin"),
                       (32, 8, "lin"), (30, 12, "lin"), (40, 8, "geo")):
        w = (np.linspace(1.0, 0.05, N) if spec == "lin"
             else 0.99 ** np.arange(N))
        C = np.diag(w)
        G = make_gaussian(C, 20000, seed=1)
        M, pdeg_used = make_graph(C, d, 20000, seed=2)
        CG, CM = cov_of(G), cov_of(M)
        rel = float(np.abs(CG - CM).max() / np.abs(CG).max())
        relC = float(np.abs(CG - C).max() / np.abs(C).max())
        rg, rm = graph_resid(G, d), graph_resid(M, d, deg=pdeg_used)
        print(f"  {N:>4}{d:>4}{spec:>10}{rel:>20.3e}{relC:>12.3e}"
              f"{deff(G):>10.4f}{deff(M):>10.4f}{rg:>8.4f}{rm:>8.4f}")
        OUT[f"N{N}_d{d}_{spec}"] = dict(cov_rel=rel, cov_vs_C=relC,
                                        d_eff_G=deff(G), d_eff_M=deff(M),
                                        resid_G=rg, resid_M=rm)
    print("\n  判读：`||C_G-C_M||/||C||` 应为**机器精度**（两者协方差完全相同），")
    print("        `d_eff` 与任何协方差泛函也相同；")
    print("        但尾部残差天差地别：高斯-1（无解释），图测度-0（完全被解释）。")
    print("        - 对**任意** C 与**任意** d，`d*` 都不是协方差的函数。")
    json.dump(OUT, open(os.path.join(HERE, "v83b_universal_blindness.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nsaved v83b_universal_blindness.json")


if __name__ == "__main__":
    main()
