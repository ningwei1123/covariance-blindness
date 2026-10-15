# -*- coding: utf-8 -*-
"""V46（任务①）：把锥—主角定理变成**适用域预测器**。

核心澄清（必须先说清）
  定理在 V = W_d（数据自适应，如取 V = 前 d 个 POD）时是**空洞的**：
  锥 C(W_d, alpha) 本身就是用 P_{W_d} 定义的，对 v 在该锥内 P_{W_d}v = v_V，
  故 P_{W_d} 在锥上恒单射、sigma_min ≡ 1，条件 1 > tan(alpha) 平凡成立。
  => **V 必须由动力学/物理先验指定，不能取自数据的协方差。**

预测器（可计算、可证伪）
  对先验子空间 V（dim V = d）：
      tan_alpha(V) = max_A ||u_{V⊥}|| / ||u_V||        （数据的**最坏情形**比值）
      sigma_min    = sigma_min(P_{W_d} | V)             （W_d = 前 d 个 POD 方向）
      margin       = sigma_min - tan_alpha
  定理：margin > 0  ==>  d* = d（且 dim V = d 是唯一可能的情形）。
  故 margin > 0 应预测"该 d 处有肘部"；margin <= 0 则不保证（且实测往往无肘部）。

在四个系统族上检验，V 全为**先验指定**：
  KS L=22/44/66     V = 振荡低波数 Fourier 子空间，dim = d_M
  双尺度 L96 K=8     V = 慢变量 X 张成的子空间，dim = K
  单尺度 L96 N=20    V = 低波数格点 Fourier 子空间，dim = d
  2D Kolmogorov     V = 低波数 2D Fourier 子空间，dim = d
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = {}


def osc_fourier_1d(N, L=None, dim=8):
    """一维振荡 Fourier 子空间（不含均值），维数恰好 dim。"""
    if L is None:
        L = N
    cols, x = [], np.arange(N) * (L / N)
    j = 1
    while len(cols) < dim:
        cols.append(np.sqrt(2.0 / N) * np.cos(2 * np.pi * j * x / L))
        if len(cols) < dim:
            cols.append(np.sqrt(2.0 / N) * np.sin(2 * np.pi * j * x / L))
        j += 1
    V = np.array(cols[:dim]).T
    return np.linalg.qr(V)[0][:, :dim]


def osc_fourier_2d(N, dim):
    """二维低波数 Fourier 子空间（不含均值），按 |k| 递增取，维数恰好 dim。"""
    kx = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    kk = np.sqrt(KX ** 2 + KY ** 2).ravel()
    idx = np.argsort(kk)
    cols = []
    seen = 0
    for i in idx:
        if seen >= dim:
            break
        m, n = divmod(i, N)
        if m == 0 and n == 0:
            continue
        # 实/虚部场
        ph = np.zeros((N, N), complex); ph[m, n] = 1.0
        c = np.real(np.fft.ifft2(ph + np.conj(np.roll(np.flip(ph), (1, 1))))).ravel()
        s = np.real(np.fft.ifft2((ph - np.conj(np.roll(np.flip(ph), (1, 1)))) / 1j)).ravel()
        for v in (c, s):
            if np.linalg.norm(v) > 1e-9:
                cols.append(v / np.linalg.norm(v)); seen += 1
                if seen >= dim:
                    break
    V = np.array(cols[:dim]).T
    return np.linalg.qr(V)[0][:, :dim]


def margin(X, Vb, d):
    Xc = X - X.mean(0)
    pv = Xc @ Vb
    perp = Xc - pv @ Vb.T
    ratio = np.linalg.norm(perp, axis=1) / np.maximum(np.linalg.norm(pv, axis=1), 1e-300)
    ta, ta_med = float(ratio.max()), float(np.median(ratio))
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:d]
    if Vb.shape[1] != d:
        smin, note = 0.0, f"dim V={Vb.shape[1]} != d={d}"
    else:
        M = W @ Vb
        smin, note = float(np.linalg.svd(M, compute_uv=False)[-1]), ""
    return dict(d=d, dim_V=int(Vb.shape[1]), tan_alpha=ta, tan_alpha_med=ta_med,
                sigma_min=smin, margin=smin - ta, note=note)


def main():
    print("=" * 108)
    print("任务①：适用域预测器 —— margin = sigma_min(P_W|V) - tan_alpha(V) 是否预测'有/无肘部'")
    print("=" * 108)

    rows = []
    # ---- KS
    for tag, L, N, dM in (("L22_N64", 22.0, 64, 8), ("L44_N64", 44.0, 64, 18),
                          ("L66_N72", 66.0, 72, 28)):
        f = os.path.join(HERE, f"v20_snap_{tag}.npy")
        if not os.path.exists(f):
            continue
        X = np.load(f).astype(np.float64)
        Vb = osc_fourier_1d(N, L, dM)
        r = margin(X, Vb, dM)
        r.update(system=f"KS L={L:.0f}", elbow="有（=d_M，A 级）", expect="有")
        rows.append(r)

    # ---- 双尺度 L96：V = 慢变量张成（即状态的前 K 个坐标方向）
    try:
        from v27_twoscale import ts_l96
        Xt, slow = ts_l96(K=8, J=10, c=10.0, b=10.0, dt=0.0005, T=200.0,
                          stride=10, seed=3)
        print(f"\n  [shape] 双尺度 L96: X={Xt.shape} slow={np.shape(slow)}")
        assert np.allclose(np.asarray(slow), Xt[:, :8]), "slow 应为状态的前 K 列"
        K = 8
        Vb = np.zeros((Xt.shape[1], K))
        for k in range(K):
            Vb[k, k] = 1.0                      # V = span(e_1..e_K) ⊂ R^88
        r = margin(Xt, Vb, K)
        r.update(system="双尺度 L96 K=8", elbow="无（肘部=8 为断崖假阳性，C 级）", expect="无")
        rows.append(r)
    except Exception as e:
        print(f"  双尺度 L96 跳过: {type(e).__name__}: {e}")

    # ---- 单尺度 L96
    for tag, N, d in (("L96_N20", 20, 8), ("L96_N40", 40, 12)):
        f = os.path.join(HERE, f"v21_snap_{tag}.npy")
        if not os.path.exists(f):
            continue
        X = np.load(f).astype(np.float64)
        Vb = osc_fourier_1d(N, None, min(d, N - 1))
        r = margin(X, Vb, min(d, N - 1))
        r.update(system=f"单尺度 {tag}", elbow="无（D 级）", expect="无")
        rows.append(r)

    # ---- 2D Kolmogorov
    f = os.path.join(HERE, "v28_ns2d_N64_nu0.02_Z.npy")
    if os.path.exists(f):
        Z = np.load(f).astype(np.float64)
        print(f"  [shape] 2D Kolmogorov Z={Z.shape}")
        if Z.shape[1] >= 4096:
            Zp = Z[:, :4096]
            for d in (8, 12, 16, 20):
                Vb = osc_fourier_2d(64, d)
                r = margin(Zp, Vb, d)
                r.update(system=f"2D Kolmogorov (d={d})", elbow="无（弱负面）", expect="无")
                rows.append(r)

    print("\n" + "=" * 108)
    print(f"{'系统':<24}{'d':>4}{'dim V':>7}{'tan α max':>11}{'tan α 中位':>12}"
          f"{'σ_min':>9}{'margin':>10}{'预测':>8}{'实测':>26}")
    for r in rows:
        pred = "有肘部" if r["margin"] > 0 else "无保证"
        print(f"{r['system']:<24}{r['d']:>4}{r['dim_V']:>7}{r['tan_alpha']:>11.4f}"
              f"{r['tan_alpha_med']:>12.4f}{r['sigma_min']:>9.4f}{r['margin']:>10.4f}"
              f"{pred:>8}  {r['elbow']:>26}")

    # 预测力评分
    correct = sum(1 for r in rows
                  if (r["margin"] > 0 and r["expect"] == "有")
                  or (r["margin"] <= 0 and r["expect"] == "无"))
    print(f"\n  预测正确：{correct}/{len(rows)}")
    print("  说明：margin>0 是**充分**条件 => 预测'有肘部'是硬预测；"
          "margin<=0 只是'无保证'，不必然无肘部。")
    OUT["rows"] = rows
    OUT["score"] = [correct, len(rows)]

    # ---------------- 空洞性对照--------------
    print("\n" + "=" * 108)
    print("空洞性对照：V 取**数据自适应**的前 d 个 POD 时，条件平凡成立（故必须用先验 V）")
    print("=" * 108)
    f = os.path.join(HERE, "v20_snap_L22_N64.npy")
    if os.path.exists(f) and "Xt" in dir():
        for Xx, nm in ((np.load(f).astype(np.float64), "KS L=22"), (Xt, "双尺度 L96 K=8")):
            Xc = Xx - Xx.mean(0)
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            d = 8
            Vb = Vt[:d].T
            pv = Xc @ Vb
            perp = Xc - pv @ Vb.T
            ta = float((np.linalg.norm(perp, axis=1) /
                        np.maximum(np.linalg.norm(pv, axis=1), 1e-300)).max())
            print(f"  {nm:<18}: V=W_{d} -> σ_min ≡ 1.0000, tan α={ta:.4f}, "
                  f"margin={1-ta:+.4f} -> 平凡'通过'（但 d* 实际={'8' if 'KS' in nm else '非 8'}）")
    json.dump(OUT, open(os.path.join(HERE, "v46_predictor.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\nsaved v46_predictor.json")


if __name__ == "__main__":
    main()
