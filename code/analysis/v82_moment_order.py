# -*- coding: utf-8 -*-
"""V82（方向 E）：把 P2 推广到 p 阶图 —— 「最低矩阶数」定理与数值验证。

P2（p=2）回顾
  F(z) = (z, {z_i z_j})   （线性 + 二次）
  Δ_k = h_kᵀ G⁻¹ h_k,  h_k = E[F r_k],  G = E[F Fᵀ]
  h 含三阶矩 E[z_i z_j r_k]，G 含四阶矩。

推广（E）
  设头部坐标 z ∈ R^d，尾部 r ∈ R^{N-d}。取**至多 p 次**的多项式特征
      F_p(z) = { z^α : 1 ≤ |α| ≤ p } ,   M_p = dim F_p = Σ_{j=1}^p C(d+j-1, j) .
  设尾部可写成**至多 p 次**的多项式图： r(x) = P_p(z) + e ,  e ⊥ F_p 。
  则最优（无惩罚）p 次多项式预测器对 r_k 的解释方差为
      Δ_k^{(p)} = h_k^{(p)ᵀ} (G^{(p)})⁻¹ h_k^{(p)} ,
  其中 h^{(p)} 含**至多 p+1 阶**矩，G^{(p)} 含**至多 2p 阶**矩。

两条可检验的推论
  E1  **充分性**：若图恰为 p 次，则 Δ^{(p)} 等于直接最小二乘投影的解释方差（机器精度）。
  E2  **最低阶数**：若图恰为 p 次（且真含 p 次项），则用 q = p−1 次特征时，
      解释方差为 0 —— 即**检测 p 次图至少需要 p+1 至 2p 阶矩**。
      （对 p=2：一阶（线性）特征解释不了任何尾部方差，这正是 P1。）
"""
import json, os, sys, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}


def feats(z, p):
    """至多 p 次的多项式特征（含线性项）。返回 (n, M_p)。"""
    n, d = z.shape
    cols = []
    for deg in range(1, p + 1):
        for alpha in itertools.combinations_with_replacement(range(d), deg):
            c = np.ones(n)
            for a in alpha:
                c = c * z[:, a]
            cols.append(c)
    return np.hstack([c[:, None] for c in cols]) if cols else np.zeros((n, 0))


def dim_Mp(d, p):
    from math import comb
    return sum(comb(d + j - 1, j) for j in range(1, p + 1))


def graph_R2(Z, R, p):
    """用至多 p 次特征对每个尾部坐标做投影，返回 (Δ_formula, Δ_direct, var)。"""
    F = feats(Z, p)
    G = F.T @ F / len(F)
    h = F.T @ R / len(F)
    Ginv = np.linalg.pinv(G)
    dform = np.einsum('ik,ij,jk->k', h, Ginv, h)
    coef = Ginv @ h
    ddir = ((F @ coef) ** 2).mean(0)
    var = (R ** 2).mean(0)
    return dform, ddir, var, F.shape[1]


def main():
    print("=" * 100)
    print("V82（方向 E）  p 阶图的「最低矩阶数」定理：数值验证")
    print("=" * 100)
    rng = np.random.default_rng(0)
    d, nt = 5, 40000
    z = rng.standard_normal((nt, d))
    Zc = z - z.mean(0)

    for p_true in (1, 2, 3, 4):
        # 构造一个**恰好 p_true 次**的尾部图：仅用**纯 p_true 次**单项式，
        # 不能混入低次项（否则低阶特征本就能解释一部分，混淆检验）
        F_all = feats(z, p_true)
        F_low = feats(z, p_true - 1) if p_true > 1 else np.zeros((nt, 0))
        pure = F_all[:, F_low.shape[1]:]          # 纯 p_true 次部分
        w = rng.standard_normal((pure.shape[1], 3))
        r = pure @ w
        # 去掉 r 中能被 F_true 解释的部分之外的？不需要：r 就是 F_true 的线性组合，
        # 故 p_true 次特征应能 100% 解释，而 p_true-1 次应解释 0（当 w 在新增项上有分量）
        r = r - r.mean(0)
        print(f"\n  ── 真实阶数 p = {p_true}   尾部维数 = {r.shape[1]}"
              f"   (M_{p_true} = {dim_Mp(d, p_true)})")
        print(f"     {'q(探测阶数)':>12}{'M_q':>6}{'Δ 公式':>13}{'Δ 直接':>13}"
              f"{'相对差':>11}{'解释占比':>10}")
        for q in range(1, p_true + 2):
            dform, ddir, var, Mq = graph_R2(Zc, r, q)
            f0, dd0 = float(dform[0]), float(ddir[0])
            rel = abs(f0 - dd0) / max(abs(dd0), 1e-300)
            frac = f0 / var[0] if var[0] > 0 else 0.0
            print(f"     {q:>12}{Mq:>6}{f0:>13.4e}{dd0:>13.4e}"
                  f"{rel:>11.2e}{frac*100:>9.1f}%")
            OUT[f"p{p_true}_q{q}"] = dict(M_q=Mq, delta_formula=f0,
                                          delta_direct=dd0, rel=rel, frac=frac)
        print(f"     ⇒ q={p_true} 时解释占比应为 100%；q={p_true-1} 应 ≈ 0"
              f"（最低需 {p_true+1} 至 {2*p_true} 阶矩）")

    # 用真实 KS 数据复核 p=2 的 P2
    print("\n" + "=" * 100)
    print("  真实 KS L=22 数据上的 p 阶扫描（d=8）")
    print("=" * 100)
    X = np.load(os.path.join(HERE, "v20_snap_L22_N64.npy")).astype(np.float64)
    if len(X) > 6000:
        X = X[::max(1, len(X)//6000)]
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    d = 8
    Z = Xc @ Vt[:d].T
    R = Xc - Z @ Vt[:d]
    print(f"     {'q':>4}{'M_q':>7}{'解释尾部方差占比':>18}{'相对差(公式/直接)':>22}")
    for q in (1, 2, 3, 4):
        dform, ddir, var, Mq = graph_R2(Z, R, q)
        frac = float(dform.sum() / var.sum())
        rel = float(np.abs(dform - ddir).max() / max(np.abs(ddir).max(), 1e-300))
        print(f"     {q:>4}{Mq:>7}{frac*100:>17.2f}%{rel:>22.2e}")
        OUT[f"ks_q{q}"] = dict(M_q=Mq, frac=frac, rel=rel)
    json.dump(OUT, open(os.path.join(HERE, "v82_moment_order.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nsaved v82_moment_order.json")


if __name__ == "__main__":
    main()
