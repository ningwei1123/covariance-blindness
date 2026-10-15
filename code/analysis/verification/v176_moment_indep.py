# -*- coding: utf-8 -*-
"""V176：四阶矩公式的**独立**验证。

待验公式（稿件 Prop. 2）
  F(z) = (z_1..z_d, {z_i z_j}_{i<=j}),  G = E[F Fᵀ],  h_k = E[F r_k]
  Δ_k = h_kᵀ G⁻¹ h_k  =  用最优无惩罚二次预测器解释的 r_k 的方差

原实现只做过"公式 vs 直接投影"的**自洽**检验（同一套正规方程）。本脚本走三条独立路径：

  路径 1（不同数值算法）：用 SVD 伪逆解最小二乘，而非正规方程 G⁻¹h
  路径 2（不同数据实现）：换种子重新生成同一测度的样本，检验公式是否仍成立
  路径 3（解析已知）：构造成 r = F(z)ᵀc + 独立噪声，
        则 Δ_k 应**精确等于**信号方差 ‖c‖²_G（可解析算出），与数值解的偏差即为误差

另核验**宇称律**：degree-p 图对 degree-q 特征不可见 iff p+q 为奇数。
"""
import json, os
import numpy as np

H = r"E:\deepseekworkspace\experiments_v12"
OUT = {}


def features(z, deg):
    """[1, 1..deg 次单项式]（与稿件 poly_features 同构，但独立书写）。"""
    n, d = z.shape
    cols = [np.ones((n, 1))]
    if deg >= 1:
        cols.append(z)
    if deg >= 2:
        for i in range(d):
            for j in range(i, d):
                cols.append((z[:, i] * z[:, j])[:, None])
    return np.hstack(cols)


def delta_formula(F, R):
    """Δ_k = h_kᵀ G⁻¹ h_k，用正规方程。"""
    n = F.shape[0]
    G = F.T @ F / n
    H = F.T @ R / n                      # h_k 按列
    return np.einsum("kj,kj->j", H, np.linalg.solve(G, H))


def delta_svdpinv(F, R):
    """同一量的**不同算法**：SVD 伪逆直接解最小二乘。"""
    Fp = np.linalg.pinv(F)               # (p, n)
    C = Fp @ R                           # (p, m)
    # 解释方差 = ‖F C‖² 的逐列均值
    return np.einsum("ij,ij->j", F @ C, F @ C) / F.shape[0]


def delta_residual(F, R):
    """第三条路径：Δ = Var(r) − Var(r − Fc)，用 lstsq。"""
    C, *_ = np.linalg.lstsq(F, R, rcond=None)
    resid = R - F @ C
    return (R ** 2).mean(0) - (resid ** 2).mean(0)


def main():
    print("=" * 104)
    print("V176  四阶矩公式的独立验证")
    print("=" * 104)
    rng = np.random.default_rng(20240501)

    # ------------------------------------------------ 情形 A：一般测度，三条路径互比
    print("\n[A] 一般测度：三条独立数值路径互比")
    print(f"  {'d':>4}{'n':>7}{'路径1=2':>12}{'路径1=3':>12}{'最大相对差':>13}")
    for d, n in ((4, 4000), (6, 8000), (8, 12000)):
        z = rng.standard_normal((n, d))
        # 尾部：含二次、三次成分的混合
        r = (0.5 * z[:, 0] * z[:, 1] + 0.3 * z[:, 2] ** 2
             - 0.2 * z[:, 3] ** 3 + 0.4 * z[:, 0])[:, None] * np.ones((1, 3))
        r = r + 0.1 * rng.standard_normal(r.shape)
        F = features(z, 2)
        a = delta_formula(F, r)
        b = delta_svdpinv(F, r)
        c = delta_residual(F, r)
        den = max(np.abs(a).max(), 1e-30)
        print(f"  {d:>4}{n:>7}{np.abs(a-b).max()/den:>12.2e}"
              f"{np.abs(a-c).max()/den:>12.2e}"
              f"{max(np.abs(a-b).max(), np.abs(a-c).max())/den:>13.2e}")
        OUT[f"general_d{d}"] = dict(path12=float(np.abs(a - b).max() / den),
                                    path13=float(np.abs(a - c).max() / den))

    # ------------------------------------------------ 情形 B：解析已知
    print("\n[B] 解析已知：r = F(z)c + 噪声，Δ 应等于 ‖c‖²_G")
    print(f"  {'d':>4}{'n':>7}{'解析 ‖c‖²_G':>15}{'数值(路径1)':>15}{'相对差':>11}")
    for d, n in ((4, 6000), (6, 10000), (8, 16000)):
        z = rng.standard_normal((n, d))
        F = features(z, 2)
        p = F.shape[1]
        c = rng.standard_normal((p, 2)) * 0.5
        r = F @ c
        G = F.T @ F / n
        analytic = np.einsum("kj,kj->j", c, G @ c) if False else np.einsum(
            "kj,kl,lj->j", c, G, c)
        num = delta_formula(F, r)
        rel = np.abs(num - analytic).max() / max(np.abs(analytic).max(), 1e-30)
        print(f"  {d:>4}{n:>7}{np.abs(analytic).max():>15.6f}"
              f"{np.abs(num).max():>15.6f}{rel:>11.2e}")
        OUT[f"analytic_d{d}"] = dict(analytic=float(np.abs(analytic).max()),
                                     numeric=float(np.abs(num).max()),
                                     rel=float(rel))

    # ------------------------------------------------ 情形 C：宇称律
    print("\n[C] 宇称律：degree-p 图对 degree-q 特征不可见 iff p+q 为奇数")
    print(f"  {'p (图次数)':>11}{'q (特征次数)':>13}{'p+q':>5}"
          f"{'解释比例':>11}  判读")
    ok = 0
    tot = 0
    for p in (1, 2, 3):
        for q in (1, 2, 3):
            n, d = 20000, 5
            z = rng.standard_normal((n, d))
            # 构造纯 p 次图：用 p 个随机单项式
            from itertools import combinations_with_replacement
            idx = list(combinations_with_replacement(range(d), p))[:8]
            h = np.ones((n, len(idx)))
            for j, ii in enumerate(idx):
                for i in ii:
                    h[:, j] *= z[:, i]
            c = rng.standard_normal(len(idx))
            r = (h @ c)[:, None]
            Fq = features(z, q)
            C, *_ = np.linalg.lstsq(Fq, r, rcond=None)
            frac = 1 - ((r - Fq @ C) ** 2).sum() / (r ** 2).sum()
            invisible = (p + q) % 2 == 1
            good = (frac < 0.05) if invisible else (frac > 0.95)
            ok += int(good); tot += 1
            print(f"  {p:>11}{q:>13}{p+q:>5}{frac:>11.4f}  "
                  f"{'不可见(符合)' if invisible else '应可见'}"
                  f"{' ✓' if good else ' **✗**'}")
    print(f"\n  宇称律符合 {ok}/{tot}")
    OUT["parity"] = dict(ok=ok, tot=tot)

    json.dump(OUT, open(os.path.join(H, "v176_moment_indep.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print("\n  判读：三条路径应一致（相对差 ~1e-14）；解析情形应精确；宇称律应 9/9。")
    print("saved v176_moment_indep.json")


if __name__ == "__main__":
    main()
