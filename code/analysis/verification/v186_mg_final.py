# -*- coding: utf-8 -*-
"""V186：用**正确的不稳定模数**重做 V183 的稳健性判定（定案）。

V183 的错误
  用 `sum(1 for r in roots if r.real>0)` 计数，而 Newton 从 14 个初值出发会
  **重复收敛到同一根**（τ=17 时 +0.0413+0.1391i 被数了 6 次），
  于是 d 被抬到 14（正确值 4），子空间吞掉吸引子，R→0，四条延迟全部"通过"。

本版
  直接用 v87 的 unstable_modes 取**正确**的不稳定模数，
  在 v88 协议下对每条延迟跑多条独立轨迹，判定 "τ=23 唯一" 是否稳健。
"""
import json, os, sys
import numpy as np

H = r"E:\deepseekworkspace\experiments_v12"
sys.path.insert(0, H)
from v87_dde_search import integrate_mg, unstable_modes
from v50_graph_slope import graph_slope
from v181_mg_fixed import v_cheb
from v182_protocol_compare import v88_fourier

OUT = {}


def main():
    print("=" * 104)
    print("V186  Mackey–Glass 「τ=23 唯一」的稳健性（正确不稳定模数）")
    print("=" * 104)
    M = 64
    for tau in (17.0, 23.0, 30.0, 40.0):
        roots, xs, fp = unstable_modes(tau, M=M)
        nu = len(roots) if not isinstance(roots, int) else roots
        # v87 的 unstable_modes 返回 (roots, xs, fp)；roots 可能是根列表
        try:
            nu = len(roots)
        except TypeError:
            nu = int(roots)
        th = np.linspace(-tau, 0.0, M)
        cols = []
        for r in roots:
            v = np.exp(r * th); cols.append(v.real)
            if abs(r.imag) > 1e-8:
                cols.append(v.imag)
        Va = np.array(cols).T if cols else np.zeros((M, 1))
        d = max(2, Va.shape[1])
        excl = max(5, int(0.05 * tau / 0.01))
        print(f"\n  ── τ={tau}   不稳定模 {nu} → V 列数 {Va.shape[1]}  d={d}  excl={excl}")
        print(f"     {'seed':>5}{'n':>7}{'d_eff':>8}{'σ_min':>9}{'R':>9}"
              f"{'margin':>10}  判读")
        vals = []
        for seed in (1, 3, 7, 11, 19):
            try:
                X = integrate_mg(tau=tau, M=M, seed=seed)
            except TypeError:
                X = integrate_mg(tau=tau, M=M)
            except Exception as e:
                print(f"     {seed:>5}   失败 {type(e).__name__}"); continue
            X = np.asarray(X, dtype=np.float64)
            if len(X) < 200 or X.std() < 1e-10:
                print(f"     {seed:>5}   数据退化"); continue
            Xc = X - X.mean(0)
            s = np.linalg.svd(Xc, compute_uv=False)
            lam = s ** 2 / (s ** 2).sum()
            deff = float(1 / (lam ** 2).sum())
            Vq = np.linalg.qr(v_cheb(M, d))[0][:, :d]
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            smin = float(np.linalg.svd(Vt[:d] @ Vq, compute_uv=False)[-1])
            g = graph_slope(X, Vq, excl=excl)
            mg = smin - g["R_max"]
            vals.append(mg)
            print(f"     {seed:>5}{len(X):>7}{deff:>8.2f}{smin:>9.4f}"
                  f"{g['R_max']:>9.4f}{mg:>+10.4f}  "
                  f"{'认证' if mg > 0 else '拒绝'}")
        if vals:
            pos = sum(1 for v in vals if v > 0)
            print(f"     → margin {min(vals):+.4f} … {max(vals):+.4f}；"
                  f"**认证 {pos}/{len(vals)}**")
            OUT[f"tau={tau}"] = dict(n_unstable=nu, d=d, excl=excl,
                                     margins=vals, n_positive=pos, n=len(vals))
    json.dump(OUT, open(os.path.join(H, "v186_mg_final.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print("\n  判读：若仅 τ=23 认证，则「τ=23 唯一」成立。")
    print("saved v186_mg_final.json")


if __name__ == "__main__":
    main()
