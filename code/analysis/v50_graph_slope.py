# -*- coding: utf-8 -*-
"""V50（任务① 收尾）：正确的图斜率估计量 —— 时间排除后的 V-空间最近邻比值。

V49 的问题
  我用"随机成对、剔除 ‖Δ_V‖ 最小的 1%"来估 L。但 sup 恰在 ‖Δ_V‖ -> 0 处取到
  （图的导数），所以那个剔除正好丢掉了要测的东西，L̂ 只是**下界**。

正确估计量
  对每个查询点，在 V-空间取最近邻，但**排除时间上邻近的样本**（±T_excl），
  然后测
        R = ‖Δ_{V⊥}‖ / ‖Δ_V‖
  · 若 A 真是 V 上的 Lipschitz 图，相似 V 必然带来相似 V⊥ -> R 有界（≈ L）；
  · 若 A 不是图（相似 V 却有很不同的 V⊥），R 会极大。
  这强制检验"图性质"，而不是时间轨迹的局部光滑性。

预期
  KS: R 有界（~0.5）且 σ_min ≈ 0.99 -> margin > 0 -> 认证
  双尺度 L96: R 极大 -> margin < 0 -> 拒绝（修好 V46 的假阳性）
  单尺度 L96 / 2D Kolmogorov: R 极大 -> 拒绝
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v27_twoscale import ts_l96                              # noqa: E402
from v46_predictor import osc_fourier_1d, osc_fourier_2d     # noqa: E402

OUT = {}


def graph_slope(X, Vb, excl, nq=500, seed=0):
    """时间排除后的 V-空间最近邻斜率 R = ‖Δ_{V⊥}‖/‖Δ_V‖。"""
    Xc = X - X.mean(0)
    pv = Xc @ Vb
    perp = Xc - pv @ Vb.T
    M = len(Xc)
    rng = np.random.default_rng(seed)
    qs = rng.choice(M, size=min(nq, M), replace=False)
    Rs = []
    for q in qs:
        dV = np.linalg.norm(pv - pv[q], axis=1)
        dV[max(0, q - excl):q + excl + 1] = np.inf
        j = int(np.argmin(dV))
        dP = float(np.linalg.norm(perp[q] - perp[j]))
        Rs.append(dP / max(float(dV[j]), 1e-300))
    Rs = np.array(Rs)
    return dict(R_max=float(Rs.max()), R_p90=float(np.quantile(Rs, 0.9)),
                R_median=float(np.median(Rs)), n=len(Rs))


def sigma_min(X, Vb, d):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    if Vb.shape[1] != d:
        return 0.0
    return float(np.linalg.svd(Vt[:d] @ Vb, compute_uv=False)[-1])


def main():
    rows = []
    # KS
    for tag, L_, N, dM in (("L22_N64", 22.0, 64, 8), ("L44_N64", 44.0, 64, 18),
                           ("L66_N72", 66.0, 72, 28)):
        f = os.path.join(HERE, f"v20_snap_{tag}.npy")
        if os.path.exists(f):
            X = np.load(f).astype(np.float64)
            Vb = osc_fourier_1d(N, L_, dM)
            g = graph_slope(X, Vb, excl=100)
            rows.append(dict(system=f"KS L={L_:.0f}", d=dM, dim_V=dM,
                             sigma_min=sigma_min(X, Vb, dM), **g,
                             expect="有", elbow="有（A 级）"))
    # 双尺度 L96（快照间隔 0.005 时间单位 -> 排除 ±4000 = ±20 时间单位）
    for K in (8, 4):
        Xk, _ = ts_l96(K=K, J=10, c=10.0, b=10.0, dt=0.0005, T=60.0, stride=10, seed=3)
        Vb = np.zeros((Xk.shape[1], K))
        for k in range(K):
            Vb[k, k] = 1.0
        g = graph_slope(Xk, Vb, excl=4000)
        rows.append(dict(system=f"双尺度 L96 K={K}", d=K, dim_V=K,
                         sigma_min=sigma_min(Xk, Vb, K), **g,
                         expect="无", elbow="断崖假阳性（C 级）"))
    # 单尺度 L96（快照间隔 0.05 -> 排除 ±400 = ±20 时间单位）
    for tag, N, d in (("L96_N20", 20, 8), ("L96_N40", 40, 12)):
        f = os.path.join(HERE, f"v21_snap_{tag}.npy")
        if os.path.exists(f):
            X = np.load(f).astype(np.float64)
            Vb = osc_fourier_1d(N, None, d)
            g = graph_slope(X, Vb, excl=400)
            rows.append(dict(system=f"单尺度 {tag}", d=d, dim_V=d,
                             sigma_min=sigma_min(X, Vb, d), **g,
                             expect="无", elbow="无（D 级）"))
    # 2D Kolmogorov
    f = os.path.join(HERE, "v46_ns2d_N64_nu0.02_Y.npy")
    if os.path.exists(f):
        Y = np.load(f).astype(np.float64)
        for d in (8, 16, 24):
            Vb = osc_fourier_2d(64, d)
            g = graph_slope(Y, Vb, excl=30)
            rows.append(dict(system=f"2D Kolmogorov d={d}", d=d, dim_V=d,
                             sigma_min=sigma_min(Y, Vb, d), **g,
                             expect="无", elbow="无（弱负面）"))

    print("=" * 122)
    print("图斜率 R（时间排除后的 V-空间最近邻）→ margin = σ_min − R_max")
    print("=" * 122)
    print(f"{'系统':<22}{'d':>4}{'R 中位':>10}{'R_p90':>10}{'R 最大':>11}"
          f"{'σ_min':>9}{'margin':>11}{'预测':>9}{'实测':>20}")
    for r in rows:
        # 严谨拒绝需要 R_max <= L；用 R_p90 会削弱保证
        # （曾因此把双尺度 L96 K=8 误认证为 +0.249）
        r["margin"] = r["sigma_min"] - r["R_max"]
        r["margin_p90"] = r["sigma_min"] - r["R_p90"]
        r["pred"] = "有肘部" if r["margin"] > 0 else "无保证"
        print(f"{r['system']:<22}{r['d']:>4}{r['R_median']:>10.4f}{r['R_p90']:>10.4f}"
              f"{r['R_max']:>11.2f}{r['sigma_min']:>9.4f}{r['margin']:>11.4f}"
              f"{r['pred']:>9}{r['elbow']:>20}")
    ok = sum(1 for r in rows if (r["margin"] > 0) == (r["expect"] == "有"))
    print(f"\n  预测正确：{ok}/{len(rows)}")
    OUT["rows"] = rows
    OUT["score"] = [ok, len(rows)]
    json.dump(OUT, open(os.path.join(HERE, "v50_graph_slope.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved v50_graph_slope.json")


if __name__ == "__main__":
    main()
