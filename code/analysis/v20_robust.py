# -*- coding: utf-8 -*-
"""V20 稳健性：肘部是否依赖 (i) 具体轨迹、(ii) 二次这一特征类的选择？

R1  用**另一条独立轨迹**（seed=11）重跑 L=22 与 L=44 的肘部检测。
R2  用**三次**特征（多项式图，项数 d^3/6 增长）重跑同一数据：若肘部仍在 8，
    则它不是二次特征容量的产物。
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v20_ks_core import run                      # noqa: E402


def poly_features(z, deg=2):
    """[1, 所有 1..deg 次单项式]（含重复，按组合枚举，不归一化）。"""
    M, d = z.shape
    feats = [np.ones((M, 1))]
    from itertools import combinations_with_replacement
    for k in range(1, deg + 1):
        idx = list(combinations_with_replacement(range(d), k))
        if not idx:
            continue
        F = np.ones((M, len(idx)))
        for j, ii in enumerate(idx):
            for i in ii:
                F[:, j] *= z[:, i]
        feats.append(F)
    return np.hstack(feats)


def elbow(X, ds, deg=2, nval=2000, lam=1e-9):
    M, N = X.shape
    ntr = M - nval
    mu = X[:ntr].mean(0, keepdims=True)
    Xtr, Xte = X[:ntr] - mu, X[ntr:] - mu
    tot = np.linalg.norm(Xte)
    _, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    res, prev = {}, None
    for d in ds:
        Phi = Vt[:d]
        ztr, zte = Xtr @ Phi.T, Xte @ Phi.T
        Rtr = Xtr - ztr @ Phi
        Ftr, Fte = poly_features(ztr, deg), poly_features(zte, deg)
        C = np.linalg.solve(Ftr.T @ Ftr + lam * np.eye(Ftr.shape[1]), Ftr.T @ Rtr)
        w = float(np.linalg.norm(Xte - (zte @ Phi + Fte @ C)) / tot)
        res[d] = dict(w=w, ratio=float(w / prev) if prev else float("nan"),
                      n_terms=int(Ftr.shape[1]))
        prev = w
    ds_l = sorted(res)
    def robust(d):
        nb = [res[k]["ratio"] for k in ds_l if k != d and abs(k - d) <= 6
              and (k - d) % 2 == 0 and not np.isnan(res[k]["ratio"])]
        return res[d]["ratio"] / float(np.median(nb)) if len(nb) >= 3 else float("nan")
    cand = [(d, robust(d)) for d in ds_l if not np.isnan(robust(d))]
    el = min(cand, key=lambda t: t[1]) if cand else (None, float("nan"))
    return res, el


def main():
    out = {}
    for tag, L, N, dt, T, nlyap in (("L22_seed11", 22.0, 64, 0.1, 6000.0, 16),
                                    ("L44_seed11", 44.0, 64, 0.1, 6000.0, 28)):
        print(f"\n### 生成独立轨迹 {tag} (seed=11)", flush=True)
        r = run(L=L, N=N, dt=dt, T=T, nlyap=nlyap, stride=5, warm=1000.0, seed=11)
        X = r["X"]
        np.save(os.path.join(HERE, f"v20_snap_{tag}.npy"), X.astype(np.float32))
        for deg, ds in ((2, list(range(1, min(2 * int(L / 3.2) + 6, 30)))),
                        (3, list(range(3, 15)) if L == 22.0 else [])):
            if not ds:
                continue
            res, el = elbow(X, ds, deg=deg)
            key = f"{tag}_deg{deg}"
            out[key] = dict(elbow_d=el[0], elbow_stat=float(el[1]),
                            results={str(k): v for k, v in res.items()})
            print(f"  [deg={deg}] 肘部 d = {el[0]} (稳健统计 {el[1]:.3f})")
            print("     d   : " + " ".join(f"{d:>8d}" for d in ds[:16]))
            print("    w_qua: " + " ".join(f"{res[d]['w']:>8.2e}" for d in ds[:16]))
            print("    比值 : " + " ".join(f"{res[d]['ratio']:>8.3f}" for d in ds[:16]))
            print("    项数 : " + " ".join(f"{res[d]['n_terms']:>8d}" for d in ds[:16]))

    json.dump(out, open(os.path.join(HERE, "v20_robust.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n===== 汇总 =====")
    for k, v in out.items():
        print(f"  {k:<18} 肘部 d = {v['elbow_d']}  (稳健统计 {v['elbow_stat']:.3f})")
    print("saved v20_robust.json")


if __name__ == "__main__":
    main()
