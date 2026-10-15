# -*- coding: utf-8 -*-
"""V85（方向 B）：降维模型实验 —— 按 d* 定尺寸 vs 按方差判据 r(eps) 定尺寸。

主张要兑现的那句话
  `r(1e-3)/d_M = 2.50 / 2.33 / 2.25` —— 标准方差判据为达到可比精度**多要 2.3-2.5 倍模态**。
  本节把它变成一个**可量化的误差-模态数曲线**，并给出"同精度下省多少模态"。

两个可比的近似
  (a) **截断**（标准做法）：`û = Φ_d Φ_dᵀ u`，误差 = 被丢弃的尾部。
  (b) **图闭合**（本文做法）：`û = Φ_d z + h(z)`，`h` 为二次图回归（闭式、免训练）。
      误差 = 图残差。

对每个 `d` 测相对状态误差，然后回答两个问题
  Q1  达到给定精度各需多少模态？（对照 2.3x）
  Q2  在 d = d* 处，图闭合相对于截断改进了多少倍？
并把结论做成"误差 vs d"曲线，供正文作图。
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = {}


def pod_basis(X, dmax):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:dmax], Xc


def graph_features(Z, deg=2):
    n, d = Z.shape
    F = [Z]
    if deg >= 2:
        for i in range(d):
            for j in range(i, d):
                F.append((Z[:, i] * Z[:, j])[:, None])
    return np.hstack(F)


def errors_vs_d(Xtr, Xte, ds, deg=2):
    """返回每个 d 的 (截断相对误差, 图闭合相对误差, 特征数)。"""
    dmax = max(ds)
    Vt, _ = pod_basis(Xtr, dmax)
    Xe = Xte - Xte.mean(0)
    denom = np.linalg.norm(Xe)
    rows = []
    for d in ds:
        P = Vt[:d]
        Z = Xe @ P.T
        # (a) 截断
        trunc = np.linalg.norm(Xe - Z @ P)
        # (b) 图闭合：在**训练集**上拟合 h，在测试集上评估
        Ztr = (Xtr - Xtr.mean(0)) @ P.T
        Rtr = (Xtr - Xtr.mean(0)) - Ztr @ P
        Ftr = graph_features(Ztr, deg)
        coef, *_ = np.linalg.lstsq(Ftr, Rtr, rcond=None)
        Fte = graph_features(Z, deg)
        close = np.linalg.norm(Xe - (Z @ P + Fte @ coef))
        rows.append((d, trunc / denom, close / denom, Ftr.shape[1]))
    return rows


def main():
    print("=" * 104)
    print("V85（方向 B）  按 d* 定尺寸 vs 按方差判据定尺寸")
    print("=" * 104)
    for tag, dM in (("L22_N64", 8), ("L44_N64", 18), ("L66_N72", 28)):
        f = os.path.join(HERE, f"v20_snap_{tag}.npy")
        if not os.path.exists(f):
            continue
        X = np.load(f).astype(np.float64)
        n = len(X)
        ntr = int(0.7 * n)
        Xtr, Xte = X[:ntr], X[ntr:]
        ds = list(range(2, min(40, X.shape[1] - 1)))
        rows = errors_vs_d(Xtr, Xte, ds)
        print(f"\n  ── KS {tag}（d_M = {dM}，训练 {ntr} / 测试 {n - ntr}）")
        print(f"     {'d':>4}{'截断误差':>13}{'图闭合误差':>13}{'改进倍数':>10}"
              f"{'特征数':>8}")
        for d, t, c, m in rows:
            if d in (4, 6, dM, dM + 4, 20, 24, 28, 32):
                print(f"     {d:>4}{t:>13.4e}{c:>13.4e}{t/max(c,1e-300):>10.1f}{m:>8}")
        # 达到截断在 d_M 处精度所需的模态数
        t_at_dM = next(t for d, t, c, m in rows if d == dM)
        need = next((d for d, t, c, m in rows if t <= t_at_dM), ds[-1])
        c_at_dM = next(c for d, t, c, m in rows if d == dM)
        # 图闭合达到同一精度所需
        need_c = next((d for d, t, c, m in rows if c <= t_at_dM), ds[-1])
        print(f"     截断在 d={dM} 的相对误差 = {t_at_dM:.4e}")
        print(f"       → 图闭合达到**同样**精度只需 d = {need_c}"
              f"（省 {need - need_c if need else 0} 个模态，比值 {need/max(need_c,1):.2f}x）")
        print(f"       → 只用截断则需 d = {need}")
        OUT[tag] = dict(d_M=dM, trunc_at_dM=t_at_dM, close_at_dM=c_at_dM,
                        d_for_same_acc_trunc=need, d_for_same_acc_close=need_c,
                        ratio=need / max(need_c, 1),
                        rows=[dict(d=d, trunc=t, close=c, M=m) for d, t, c, m in rows])

    json.dump(OUT, open(os.path.join(HERE, "v85_rom_sizing.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nsaved v85_rom_sizing.json")


if __name__ == "__main__":
    main()
