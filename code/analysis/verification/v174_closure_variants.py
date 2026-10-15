# -*- coding: utf-8 -*-
"""V174：闭包协议差异的受控对照 —— 判定 6.3× vs 4.6× 的来源。

四个变体（其余条件严格相同）
  A: v85 原协议        无截距 + lstsq + 测试集自居中
  B: 本审计实现        有截距 + ridge(1e-9) + 训练均值居中
  C: 无截距 + ridge + 训练均值居中
  D: 有截距 + lstsq + 训练均值居中
"""
import json, os
import numpy as np

H = r"E:\deepseekworkspace\experiments_v12"
OUT = {}


def feats(Z, intercept):
    n, d = Z.shape
    cols = [np.ones((n, 1)), Z] if intercept else [Z]
    for i in range(d):
        for j in range(i, d):
            cols.append((Z[:, i] * Z[:, j])[:, None])
    return np.hstack(cols)


def solve(F, R, ridge):
    if ridge:
        return np.linalg.solve(F.T @ F + ridge * np.eye(F.shape[1]), F.T @ R)
    c, *_ = np.linalg.lstsq(F, R, rcond=None)
    return c


def run(X, ds, intercept, ridge, test_self_center, dM):
    n = len(X)
    ntr = int(n * 0.8)
    Xtr, Xte = X[:ntr], X[ntr:]
    mu = Xtr.mean(0, keepdims=True)
    Xtr_c = Xtr - mu
    Xte_c = (Xte - Xte.mean(0)) if test_self_center else (Xte - mu)
    _, _, Vt = np.linalg.svd(Xtr_c, full_matrices=False)
    denom = np.linalg.norm(Xte_c)
    out = {}
    for d in ds:
        P = Vt[:d]
        Ztr, Zte = Xtr_c @ P.T, Xte_c @ P.T
        Rtr = Xtr_c - Ztr @ P
        C = solve(feats(Ztr, intercept), Rtr, ridge)
        trunc = np.linalg.norm(Xte_c - Zte @ P) / denom
        close = np.linalg.norm(Xte_c - (Zte @ P + feats(Zte, intercept) @ C)) / denom
        out[d] = (float(trunc), float(close))
    return out


def main():
    print("=" * 104)
    print("V174  闭包协议差异的受控对照")
    print("=" * 104)
    files = {22: ("v20_snap_L22_N64.npy", 8), 44: ("v20_snap_L44_N64.npy", 18),
             66: ("v20_snap_L66_N72.npy", 28)}
    VAR = [("A v85原协议", False, None, True),
           ("B 本审计", True, 1e-9, False),
           ("C 无截距+ridge", False, 1e-9, False),
           ("D 有截距+lstsq", True, None, False)]
    print(f"  {'变体':<18}" + "".join(f"{'L='+str(L):>26}" for L in files))
    print("  " + "-" * 96)
    for name, ic, rg, sc in VAR:
        line = f"  {name:<18}"
        for L, (fn, dM) in files.items():
            p = os.path.join(H, fn)
            if not os.path.exists(p):
                line += f"{'缺':>26}"; continue
            X = np.asarray(np.load(p, mmap_mode="r"), dtype=np.float64)
            ds = list(range(2, min(45, X.shape[1] - 1)))
            r = run(X, ds, ic, rg, sc, dM)
            t, c = r[dM]
            line += f"{f'{t:.4f}->{c:.4f} ({t/c:.2f}x)':>26}"
            OUT.setdefault(name, {})[f"L={L}"] = dict(trunc=t, close=c,
                                                      improv=t / c)
        print(line)
    json.dump(OUT, open(os.path.join(H, "v174_closure_variants.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print("\n  参考：v85 报 4.97/4.66/4.58×；本审计（变体 B）报 6.28/6.58/6.93×")
    print("saved v174_closure_variants.json")


if __name__ == "__main__":
    main()
