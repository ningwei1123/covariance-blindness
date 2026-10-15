# -*- coding: utf-8 -*-
"""V20 实验四：无模型假设的嵌入/单射维数检验（伪最近邻，FNN）。

问题（与 2025 可观测性论文的设问完全一致）：
   用「前 m 个 POD 坐标」作为 m 个观测量时，状态是否被唯一确定？
   最小可行的 m 称为该测量集下的**嵌入维数**。

方法（Kennel-Brown-Abarbanel 伪最近邻，用 POD 坐标替代延迟坐标）：
   对每个样本 i，在 d 维投影空间里找最近邻 j（排除时间近邻）；
   若全空间距离 ||u_i - u_j|| 远大于全空间最近邻的典型距离，则 j 是「伪近邻」。
   伪近邻比例为 0 的最小 d = 嵌入维数 d*。

该方法不含任何模型类（不是自编码器、不是二次图、不是阈值截断），
因此可作为二次图肘部 / 自编码器拐点的独立裁判。

同时给出吸引子的**局部内蕴维数**估计（局部 PCA 谱），用于对照 D_KY 与 d_M。
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def fnn_curve(X, ds, n_query=2000, exclude=20, seed=0, scale_quantile=0.5):
    """返回每个 d 的伪近邻比例，以及诊断量。"""
    M, N = X.shape
    rng = np.random.default_rng(seed)
    q = rng.choice(M, size=min(n_query, M), replace=False)
    # 全空间距离矩阵（分块以省内存）
    def dist_full(idx):
        # idx: (n,) 查询下标 -> (n, M)
        A = X[idx]
        return np.sqrt(np.maximum(
            (A ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None, :] - 2 * A @ X.T, 0.0))
    D = dist_full(q)
    for a, i in enumerate(q):
        lo, hi = max(0, i - exclude), min(M, i + exclude + 1)
        D[a, lo:hi] = np.inf
    full_nn = D.min(axis=1)
    ref = float(np.quantile(full_nn, scale_quantile))
    # POD 基
    U, s, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    out = {}
    for d in ds:
        Z = X @ Vt[:d].T                      # (M, d)
        Zq = Z[q]
        Dz = np.sqrt(np.maximum(
            (Zq ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None, :] - 2 * Zq @ Z.T, 0.0))
        for a, i in enumerate(q):
            lo, hi = max(0, i - exclude), min(M, i + exclude + 1)
            Dz[a, lo:hi] = np.inf
        j = Dz.argmin(axis=1)
        d_full = np.linalg.norm(X[q] - X[j], axis=1)          # 该邻居在全空间的距离
        ratio = d_full / np.maximum(full_nn, 1e-30)
        frac = float(np.mean(ratio > 10.0))
        out[int(d)] = dict(frac_false=frac,
                           median_ratio=float(np.median(ratio)),
                           p90_ratio=float(np.quantile(ratio, 0.9)),
                           ref_nn_scale=ref)
    return out, dict(full_nn_median=float(np.median(full_nn)), ref=ref)


def local_dim(X, n_query=800, k=40, seed=0, rel_thresh=0.05):
    """局部 PCA 内蕴维数估计：每个样本用 k 个近邻做 PCA。"""
    M, N = X.shape
    rng = np.random.default_rng(seed)
    q = rng.choice(M, size=min(n_query, M), replace=False)
    dims = []
    spec = []
    for i in q:
        dist = np.linalg.norm(X - X[i], axis=1)
        dist[max(0, i - 20):i + 21] = np.inf
        nb = np.argsort(dist)[:k]
        Y = X[nb] - X[nb].mean(0)
        sv = np.linalg.svd(Y, compute_uv=False)
        sv = sv / sv[0]
        d_loc = int(np.sum(sv > rel_thresh))
        dims.append(d_loc)
        spec.append(sv[:12])
    return dict(dim_median=float(np.median(dims)), dim_mean=float(np.mean(dims)),
                dim_hist={str(v): int(np.sum(np.array(dims) == v)) for v in np.unique(dims)},
                spec_median=[float(v) for v in np.median(np.array(spec), axis=0)])


def l96(N, F=8.0, dt=0.01, n=20000, sample=5, seed=1, warm=5000):
    x = F * np.ones(N) + 0.01 * np.random.default_rng(seed).standard_normal(N)

    def f(u):
        return np.roll(u, -1) * (np.roll(u, 1) - np.roll(u, -2)) - u + F

    for _ in range(warm):
        k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    out = []
    for i in range(n):
        k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if i % sample == 0:
            out.append(x.copy())
    return np.asarray(out)


def make_aligned_graph(d0, N=64, n=20000, amp=0.05, kind="quad", decay=1.3, seed=0):
    """严格「前 d0 个 POD 方向上的图」，且非线性项足够小以保证对齐。"""
    rng = np.random.default_rng(seed)
    Z = l96(d0, n=n, sample=5, seed=seed)
    Z = (Z - Z.mean(0)) / Z.std(0)
    Z = Z * (decay ** (-np.arange(d0)))
    Phi, _ = np.linalg.qr(rng.standard_normal((N, d0)))
    iu = np.triu_indices(d0)
    if kind == "quad":
        H = rng.standard_normal((N, len(iu[0])))
        H = H - Phi @ (Phi.T @ H)
        hz = (Z[:, iu[0]] * Z[:, iu[1]]) @ H.T
    else:
        H = rng.standard_normal((N, d0))
        H = H - Phi @ (Phi.T @ H)
        hz = np.tanh(2 * Z) @ H.T
    hz = hz / np.linalg.norm(hz) * amp * np.linalg.norm(Z @ Phi.T)
    return Z @ Phi.T + hz, Phi


def main():
    out = {}
    eps_target = 0.02   # 伪近邻比例阈值
    print("=" * 104)
    print("A) KS 三个域长：用前 d 个 POD 坐标作观测量时的嵌入维数（FNN）")
    print("=" * 104)
    dM = {"L22_N64": 8, "L44_N64": 18, "L66_N192": 28}
    dsets = {"L22_N64": list(range(1, 21)), "L44_N64": list(range(2, 41, 2)),
             "L66_N192": list(range(4, 49, 2))}
    for tag in ["L22_N64", "L44_N64", "L66_N192"]:
        X = np.load(os.path.join(HERE, f"v20_snap_{tag}.npy")).astype(np.float64)
        ds = dsets[tag]
        fnn, info = fnn_curve(X, ds)
        ld = local_dim(X)
        dstar = None
        for d in ds:
            if fnn[d]["frac_false"] <= eps_target:
                dstar = d
                break
        out[tag] = dict(fnn=fnn, info=info, local_dim=ld, d_star=dstar, d_M_lit=dM[tag])
        print(f"\n[{tag}]  N={X.shape[1]}  文献 d_M={dM[tag]}  局部内蕴维数中位={ld['dim_median']:.0f}")
        print("   d    : " + " ".join(f"{d:>7d}" for d in ds[:20]))
        print("   伪近邻: " + " ".join(f"{fnn[d]['frac_false']:>7.3f}" for d in ds[:20]))
        print("   中位比: " + " ".join(f"{fnn[d]['median_ratio']:>7.2f}" for d in ds[:20]))
        print(f"   >> 伪近邻比例 <= {eps_target} 的最小 d = {dstar}")

    print("\n" + "=" * 104)
    print("B) 对照：真值已知")
    print("=" * 104)
    # B1 线性高斯秩 10
    rng = np.random.default_rng(7)
    Uq, _ = np.linalg.qr(rng.standard_normal((64, 10)))
    Xg = rng.standard_normal((20000, 10)) * (0.85 ** np.arange(10)) @ Uq.T
    fnn, info = fnn_curve(Xg, list(range(1, 16)))
    ds_ = [d for d in range(1, 16) if fnn[d]["frac_false"] <= eps_target]
    out["ctrl_linear_rank10"] = dict(fnn=fnn, d_star=(ds_[0] if ds_ else None), expected=10)
    print(f"[线性高斯 秩10]  期望 10  测得 d* = {ds_[0] if ds_ else None}")
    print("   伪近邻: " + " ".join(f"{fnn[d]['frac_false']:>7.3f}" for d in range(1, 16)))

    # B2/B3 对齐的合成图流形
    for d0 in (5, 8, 12):
        Xs, Phi = make_aligned_graph(d0, seed=10 + d0)
        # 验证对齐：前 d0 个 POD 方向与 span(Phi) 的主角
        _, _, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
        C = Vt[:d0] @ Phi
        ang = np.degrees(np.arccos(np.clip(np.linalg.svd(C, compute_uv=False), -1, 1)))
        fnn, info = fnn_curve(Xs, list(range(1, d0 + 6)))
        ds_ = [d for d in range(1, d0 + 6) if fnn[d]["frac_false"] <= eps_target]
        out[f"ctrl_graph_d{d0}"] = dict(fnn=fnn, d_star=(ds_[0] if ds_ else None), expected=d0,
                                        align_max_angle_deg=float(ang.max()))
        print(f"[对齐合成图 d0={d0}]  期望 {d0}  测得 d* = {ds_[0] if ds_ else None}  "
              f"(前 d0 个 POD 方向与真图子空间最大主角 {ang.max():.2f}°)")
        print("   伪近邻: " + " ".join(f"{fnn[d]['frac_false']:>7.3f}" for d in range(1, d0 + 6)))

    json.dump(out, open(os.path.join(HERE, "v20_injectivity.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n===== 汇总 =====")
    for k, v in out.items():
        print(f"  {k:<22} 期望={str(v.get('expected', v.get('d_M_lit'))):>4}  测得 d*={v['d_star']}")
    print("\nsaved v20_injectivity.json")


if __name__ == "__main__":
    main()
