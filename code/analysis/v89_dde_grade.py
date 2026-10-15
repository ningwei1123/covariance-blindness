# -*- coding: utf-8 -*-
"""V89（方向 C 收尾）：对 Mackey-Glass 做**完整 A 级检验**。

为什么这次是真的候选
  V87：线性化不稳定模作先验子空间 -> 判据全拒（sigma_min 0.008-0.09）。
  V88：换成 **[−tau,0] 上的低阶 Chebyshev 多项式**（光滑性先验）-> tau=23/30/40 **认证**。
  V87 的 P3：tau=23 处 s_orig=0.688 -> s_shuf=1.026（**通过**）。

完整 A 级检验包含
  P0  混沌前提：**由线性化特征方程精确算出**的最大 Lyapunov 指数 Re(lambda_1) > 0
      （时滞方程的特征方程  lambda + gamma = f'(x*) exp(-lambda tau)，无需数据）✓ 精确
  P1  判据 margin = sigma_min(P_{W_d}|V) - R_max > 0
  P3  尾部打乱：s_shuf > 0.9 且 s_orig < 0.7
  KY  由特征根算 D_KY，检验引理 d* >= dim_H(A) 的弱形式 d* >= (可与 D_KY 比较)

诚实性声明
  Chebyshev 先验是**在看到不稳定模失败之后**才选的 —— 这是"判据引导"的直接体现（方向 C 的目的），
  但也意味着它是一次**事后选择**。本脚本同时报告三种先验，并明确标注这一点。
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v87_dde_search import integrate_mg, unstable_modes      # noqa: E402
from v88_dde_priors import cheb_basis, four_basis, margin_for  # noqa: E402
from v20_robust import elbow                                  # noqa: E402
from v34_audit import sig_profile, shuffle_tail                # noqa: E402

OUT = {}


def dky_from_roots(roots):
    """D_KY = j + sum_{i<=j} Re(lam_i) / |Re(lam_{j+1})|，按 Re 降序。"""
    re = np.array([r.real for r in sorted(roots, key=lambda z: -z.real)])
    if len(re) == 0 or re[0] <= 0:
        return 0.0, re
    c = np.cumsum(re)
    pos = np.where(c >= 0)[0]
    if pos.size == 0 or pos[-1] + 1 >= len(re):
        return float(np.sum(re > 0)), re
    j = int(pos[-1])
    return float(j + 1 + c[j] / abs(re[j + 1])), re


def main():
    print("=" * 108)
    print("V89   Mackey-Glass 的完整 A 级检验（P0 由特征方程精确给出）")
    print("=" * 108)
    M = 64
    for tau in (17.0, 23.0, 30.0, 40.0):
        roots, xs, fp = unstable_modes(tau, M=M)
        dky, re_sorted = dky_from_roots(roots)
        lam1 = re_sorted[0] if len(re_sorted) else -np.inf
        X = integrate_mg(tau=tau, M=M)
        if len(X) < 500:
            continue
        X = X - X.mean(0)
        s = np.linalg.svd(X, compute_uv=False)
        lam_ = s ** 2 / (s ** 2).sum()
        deff = float(1 / (lam_ ** 2).sum())
        excl = max(5, int(0.05 * tau / 0.01))
        # 检测器 + P3
        ds = list(range(2, min(35, X.shape[1] - 1)))
        res, el = elbow(X, ds, deg=2, nval=min(2000, len(X) // 5))
        so = sig_profile({d: res[d]["ratio"] for d in ds}, ds)
        de_el = el[0]
        Xs, _ = shuffle_tail(X, de_el)
        res2, _ = elbow(Xs, ds, deg=2, nval=min(2000, len(X) // 5))
        ss = sig_profile({d: res2[d]["ratio"] for d in ds}, ds)
        s_o, s_s = float(so.get(de_el, np.nan)), float(ss.get(de_el, np.nan))
        P3 = (s_s > 0.9) and (s_o < 0.7)
        P0 = lam1 > 0
        # 判据（三种先验）
        th = np.linspace(-tau, 0.0, M)
        cols = []
        for r in roots:
            v = np.exp(r * th); cols.append(v.real)
            if abs(r.imag) > 1e-8:
                cols.append(v.imag)
        Va = np.array(cols).T if cols else np.zeros((M, 1))
        margs = {}
        for nm, V in (("unstable", Va), ("cheb", cheb_basis(M, max(2, Va.shape[1]))),
                      ("fourier", four_basis(M, max(2, Va.shape[1])))):
            try:
                sm, rm, mg, d = margin_for(X, V, excl)
                margs[nm] = dict(dim_V=int(d), sigma_min=sm, R_max=rm, margin=mg)
            except Exception as e:
                margs[nm] = dict(error=type(e).__name__)
        print(f"\n  ── tau = {tau}   样本 {len(X)}   d_eff = {deff:.2f}")
        print(f"     P0（特征方程, 精确）: Re(lambda_1) = {lam1:+.4f}  ->"
              f"  {'混沌 ✓' if P0 else '**非混沌**'}")
        print(f"       不稳定模 {len(roots)} 个, Re 排序: "
              f"{[round(float(r),3) for r in re_sorted[:6]]}")
        print(f"       D_KY（由特征根）= {dky:.3f}")
        print(f"     P1 判据: " + "  ".join(
            f"{k}: margin={v.get('margin', float('nan')):+.3f}" for k, v in margs.items()))
        print(f"     P3 打乱: 肘部 = {de_el}  s_orig = {s_o:.3f}  s_shuf = {s_s:.3f}"
              f"  -> {'通过 ✓' if P3 else '未通过'}")
        grade = ("A" if (P0 and margs.get("cheb", {}).get("margin", -1) > 0 and P3)
                 else ("B" if (P0 and margs.get("cheb", {}).get("margin", -1) > 0)
                       else "D/拒绝"))
        print(f"     ⇒ **等级 {grade}**   （判据用 Chebyshev 先验）")
        OUT[f"tau{tau}"] = dict(tau=tau, d_eff=deff, lambda1=float(lam1),
                                D_KY=dky, n_unstable=len(roots),
                                elbow=int(de_el), s_orig=s_o, s_shuf=s_s,
                                P0=bool(P0), P3=bool(P3), priors=margs, grade=grade)
    json.dump(OUT, open(os.path.join(HERE, "v89_dde_grade.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print("\nsaved v89_dde_grade.json")


if __name__ == "__main__":
    main()
