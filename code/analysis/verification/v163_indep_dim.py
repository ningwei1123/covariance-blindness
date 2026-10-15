# -*- coding: utf-8 -*-
"""V163：独立测定合成数据的**POD 基图维数**，以判定估计量是否真的有偏。

关键区分（由 V159/V160 数据揭示）
  「真值 d0」是**生成基**下的图维数；估计量返回的是 **POD 基**下的图维数。
  强非线性时 POD 基混合头尾，二者本不必相等。
  若独立测定显示 POD 基维数确实是 2·d0，则**估计量无偏，是基准的标签错了**。

独立测定（不经估计量的肘部逻辑）
  d_true = min { d : w(d) < 1e-12 }，即尾部在 POD 头坐标下**严格**成为函数的最小维数。
  同时报告 w 的数值噪声底，以判断 1e-12 是否高于噪声。
"""
import json, os, sys
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import v20_robust
from v84_baselines import make_graph_data

OUT = {}


def w_profile(X, ds, deg=2, nval=800):
    M, N = X.shape
    ntr = M - nval
    mu = X[:ntr].mean(0, keepdims=True)
    Xtr, Xte = X[:ntr] - mu, X[ntr:] - mu
    tot = np.linalg.norm(Xte)
    _, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    w = {}
    for d in ds:
        Phi = Vt[:d]
        ztr, zte = Xtr @ Phi.T, Xte @ Phi.T
        Rtr = Xtr - ztr @ Phi
        Ftr = v20_robust.poly_features(ztr, deg)
        Fte = v20_robust.poly_features(zte, deg)
        C = np.linalg.solve(Ftr.T @ Ftr + 1e-9 * np.eye(Ftr.shape[1]), Ftr.T @ Rtr)
        w[d] = float(np.linalg.norm(Xte - (zte @ Phi + Fte @ C)) / tot)
    return w


def main():
    print("=" * 104)
    print("V163  合成数据的 POD 基图维数（独立测定）vs 估计量返回")
    print("=" * 104)
    print(f"  {'case':<20}{'amp':>7}{'估计量':>8}{'独立测定':>9}{'w(5)':>10}"
          f"{'w(2d0)':>10}{'噪声底':>10}  判读")

    for d0 in (5, 8, 12):
        for amp in (0.001, 0.005, 0.02, 0.1, 0.4):
            X = np.asarray(make_graph_data(d0, 40, 6000, amp=amp, seed=d0), np.float64)
            ds = list(range(2, min(30, X.shape[1] - 1)))
            _, el = v20_robust.elbow(X, ds, deg=2, nval=800)
            w = w_profile(X, ds, deg=2, nval=800)
            # 独立测定：严格成为函数的 d
            strict = [d for d in ds if w[d] < 1e-12]
            d_ind = min(strict) if strict else None
            floor = min(w.values())
            verdict = ""
            if d_ind is not None:
                if el[0] == d_ind:
                    verdict = "**一致 → 估计量无偏，是标签错了**"
                else:
                    verdict = f"不一致（差 {el[0]-d_ind:+d}）"
            print(f"  d0={d0:<16}{amp:>7.3f}{el[0]:>8}{str(d_ind):>9}"
                  f"{w[5]:>10.2e}{w.get(2*d0, float('nan')):>10.2e}"
                  f"{floor:>10.2e}  {verdict}")
            OUT[f"d0={d0}_amp={amp}"] = dict(elbow=int(el[0]), indep=d_ind,
                                             w5=float(w[5]), floor=float(floor))

    json.dump(OUT, open(os.path.join(H, "v163_indep_dim.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print("\n  判读：若「独立测定」始终等于估计量返回，则估计量无偏，问题在基准标签；")
    print("        若二者不等，则为估计量的真实偏差。")
    print("saved v163_indep_dim.json")


if __name__ == "__main__":
    main()
