# -*- coding: utf-8 -*-
"""V175：用**一致协议**复算「匹配精度下的节省」，并修掉 L=66 的搜索范围。"""
import json, os
import numpy as np

H = r"E:\deepseekworkspace\experiments_v12"
OUT = {}


def feats(Z):
    n, d = Z.shape
    cols = [np.ones((n, 1)), Z]
    for i in range(d):
        for j in range(i, d):
            cols.append((Z[:, i] * Z[:, j])[:, None])
    return np.hstack(cols)


def curves(X, ds):
    n = len(X); ntr = int(n * 0.8)
    Xtr, Xte = X[:ntr], X[ntr:]
    mu = Xtr.mean(0, keepdims=True)          # 训练均值，训练与测试统一使用
    Xtr_c, Xte_c = Xtr - mu, Xte - mu
    _, _, Vt = np.linalg.svd(Xtr_c, full_matrices=False)
    den = np.linalg.norm(Xte_c)
    T, C = {}, {}
    for d in ds:
        P = Vt[:d]
        Ztr, Zte = Xtr_c @ P.T, Xte_c @ P.T
        Rtr = Xtr_c - Ztr @ P
        Ftr, Fte = feats(Ztr), feats(Zte)
        co = np.linalg.solve(Ftr.T @ Ftr + 1e-9 * np.eye(Ftr.shape[1]), Ftr.T @ Rtr)
        T[d] = float(np.linalg.norm(Xte_c - Zte @ P) / den)
        C[d] = float(np.linalg.norm(Xte_c - (Zte @ P + Fte @ co)) / den)
    return T, C


def main():
    print("=" * 100)
    print("V175  匹配精度下的节省（一致协议）")
    print("=" * 100)
    files = {22: ("v20_snap_L22_N64.npy", 8), 44: ("v20_snap_L44_N64.npy", 18),
             66: ("v20_snap_L66_N72.npy", 28)}
    print(f"  {'L':>4}{'d_M':>5}{'r(1e-3)':>9}{'截断@r':>11}{'闭包需 d':>10}"
          f"{'节省':>8}{'稿件值':>9}")
    for L, (fn, dM) in files.items():
        p = os.path.join(H, fn)
        if not os.path.exists(p):
            print(f"  {L}: 缺"); continue
        X = np.asarray(np.load(p, mmap_mode="r"), dtype=np.float64)
        N = X.shape[1]
        ds = list(range(2, min(N - 1, 90)))          # 扩到 90 以覆盖 L=66
        T, C = curves(X, ds)
        rev = [d for d in ds if T[d] <= 1e-3]
        r_eps = min(rev) if rev else None
        if r_eps is None:
            print(f"  {L:>4}{dM:>5}   截断在 d<={ds[-1]} 内未达 1e-3")
            continue
        tgt = T[r_eps]
        cand = [d for d in ds if C[d] <= tgt]
        d_close = min(cand) if cand else None
        ref = {22: 1.235, 44: 1.303, 66: 1.306}[L]
        if d_close:
            print(f"  {L:>4}{dM:>5}{r_eps:>9}{tgt:>11.5f}{d_close:>10}"
                  f"{r_eps/d_close:>8.2f}{ref:>9.3f}")
            OUT[f"L={L}"] = dict(d_M=dM, r_eps=int(r_eps), trunc_at_r=float(tgt),
                                 d_close=int(d_close), saved=r_eps / d_close)
        else:
            print(f"  {L:>4}{dM:>5}{r_eps:>9}{tgt:>11.5f}{'—':>10}"
                  f"{'—':>8}{ref:>9.3f}   （闭包在该范围内未达目标精度）")
    json.dump(OUT, open(os.path.join(H, "v175_matched_indep.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print("\n  判读：一致协议下若节省 >1.3×，则稿件值 1.24–1.31× 亦偏低。")
    print("saved v175_matched_indep.json")


if __name__ == "__main__":
    main()
