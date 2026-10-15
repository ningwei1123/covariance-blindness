# -*- coding: utf-8 -*-
"""V201：L=66 的 block bootstrap 与窗口扫描（补齐摘要中 "28" 的统计与稳健性支撑）。

以前只对 L=22/44 做过 bootstrap（S8.2），L=66 未做，而 "28" 恰是摘要级卖点。
本脚本：
  ① 对 N=192 的生产数据做 20 次 block bootstrap，给出肘点分布
  ② 扫描宇称归一化窗口 w = 4..10
  ③ 同时报告 N=72 作对照
"""
import io, json, os, sys
import numpy as np

H = r"E:\deepseekworkspace\experiments_v12"
sys.path.insert(0, H)
from v20_robust import poly_features


def w_profile(X, ds, deg=2, nval=2000, lam=1e-9, seed=0):
    """返回每个 d 的相对尾残差 w(d) 与步比 ratio(d)=(w(d-1)-w(d))/w(d-1)。"""
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    ntr = n - min(nval, n // 3)
    tr, va = X[idx[:ntr]], X[idx[ntr:]]
    mu = tr.mean(0)
    tc, vc = tr - mu, va - mu
    Vt = np.linalg.svd(tc, full_matrices=False)[2]
    res = {}
    for d in ds:
        if d >= X.shape[1] - 1:
            continue
        P = Vt[:d]
        ztr, zva = tc @ P.T, vc @ P.T
        Rtr = tc - ztr @ P
        Ftr = poly_features(ztr, deg)
        G = Ftr.T @ Ftr + lam * np.eye(Ftr.shape[1])
        C = np.linalg.solve(G, Ftr.T @ Rtr)
        Rva = vc - zva @ P
        pred = poly_features(zva, deg) @ C
        res[d] = float(np.linalg.norm(Rva - pred) / (np.linalg.norm(Rva) + 1e-300))
    return res


def elbow_win(X, ds, deg=2, nval=2000, win=6, seed=0):
    res = w_profile(X, ds, deg, nval, seed=seed)
    ks = sorted(res)
    ratio = {}
    for i, d in enumerate(ks):
        if i == 0:
            ratio[d] = np.nan
        else:
            prev = res[ks[i - 1]]
            ratio[d] = (prev - res[d]) / (prev + 1e-300)

    def robust(d):
        nb = [ratio[k] for k in ks
              if k != d and abs(k - d) <= win and (k - d) % 2 == 0
              and not np.isnan(ratio[k])]
        if not nb or np.isnan(ratio.get(d, np.nan)):
            return np.nan
        return ratio[d] / np.median(nb)

    cand = [(d, robust(d)) for d in ks if not np.isnan(robust(d))]
    if not cand:
        return None, res, ratio
    best = max(cand, key=lambda t: t[1])
    return best[0], res, ratio


def block_bootstrap(X, nboot=20, block=200, ds=None, seed=1):
    rng = np.random.default_rng(seed)
    n = len(X)
    nb = max(1, n // block)
    els = []
    for b in range(nboot):
        starts = rng.integers(0, n - block, size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])
        try:
            e, _, _ = elbow_win(X[idx], ds, win=6, seed=100 + b)
            if e is not None:
                els.append(e)
        except Exception:
            pass
    return els


print("=" * 96)
print("V201  L=66 的 block bootstrap 与窗口扫描")
print("=" * 96)

for tag, f, Nres in (("L=66 N=192 (production)", "v20_snap_L66_N192.npy", 192),
                     ("L=66 N=72  (coarse)",     "v20_snap_L66_N72.npy", 72)):
    p = os.path.join(H, f)
    if not os.path.exists(p):
        print(f"\n  {tag}: 缺文件"); continue
    X = np.asarray(np.load(p, mmap_mode="r")[:8000], dtype=np.float64)
    ds = list(range(20, 38)) if Nres > 100 else list(range(18, 36))
    ds = [d for d in ds if d < X.shape[1] - 1]

    print(f"\n  ── {tag}   n={len(X)}  N={X.shape[1]}  ds={ds[0]}..{ds[-1]}")
    # ① 窗口扫描
    print(f"     窗口扫描：")
    for w in (4, 5, 6, 7, 8, 9, 10):
        e, _, _ = elbow_win(X, ds, win=w, seed=0)
        print(f"        w={w:<3} -> elbow = {e}")
    # ② block bootstrap
    els = block_bootstrap(X, nboot=20, block=200, ds=ds)
    if els:
        u, c = np.unique(els, return_counts=True)
        dist = ", ".join(f"{int(k)}:{int(v)}" for k, v in zip(u, c))
        print(f"     block bootstrap (20 次, block=200): 分布 {dist}")
        print(f"        中位数 {int(np.median(els))}   均值 {np.mean(els):.2f}   "
              f"命中 28 的比例 {100*np.mean(np.array(els)==28):.0f}%")
    else:
        print("     block bootstrap 全部失败")
