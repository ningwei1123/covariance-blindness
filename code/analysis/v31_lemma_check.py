# -*- coding: utf-8 -*-
"""V31：机理命题化的**数值检验** —— 引理（下界） d* >= D_KY。

引理（见 V31_LEMMA.md）：
  若前 d 个 POD 方向上的正交投影在吸引子 A 上是**双 Lipschitz**的，
  则 dim_H(A) <= d，于是
        D_KY(μ) <= D_1(μ) <= dim_H(A) <= d ,
  故（理想化的）奴役开维数 d*_ideal >= D_KY。

本脚本补上**唯一缺失的一环**：双尺度 Lorenz-96 的 D_KY（此前只算了肘部）。
它是本引理最强的检验对象，因为 d* = K 由构造已知（K=4/8）。
若 D_KY > K，则引理被否证，"肘部 = 奴役开维数"的解释需要修正。
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def ts_l96_jac(X, Y, K, J, F, h, c, b):
    """双尺度 L96 的 RHS 与 Jacobian（解析构造，向量化）。"""
    dX = (-np.roll(X, 1) * (np.roll(X, 2) - np.roll(X, -1)) - X + F
          - (h * c / b) * Y.sum(axis=1))
    dY = (-c * b * np.roll(Y, -1, axis=1) * (np.roll(Y, -2, axis=1) - np.roll(Y, 1, axis=1))
          - c * Y + (h * c / b) * X[:, None])
    # --- Jacobian ---
    n = K + K * J
    A = np.zeros((n, n))
    # dX/dX
    A[np.arange(K), np.arange(K)] += -1.0
    A[np.arange(K), (np.arange(K) - 1) % K] += -(np.roll(X, 2) - np.roll(X, -1))
    A[np.arange(K), (np.arange(K) - 2) % K] += -np.roll(X, 1)
    A[np.arange(K), (np.arange(K) + 1) % K] += np.roll(X, 1)
    # dX/dY : 每个 k 的所有 j 列
    Yidx = K + np.arange(K)[:, None] * J + np.arange(J)[None, :]      # (K, J)
    for k in range(K):
        A[k, Yidx[k, :]] += -(h * c / b)
    # dY/dX 与 dY/dY
    for k in range(K):
        rows = Yidx[k, :]
        A[rows, k] += (h * c / b)
        A[rows, rows] += -c
        A[rows, Yidx[k, (np.arange(J) + 1) % J]] += -c * b * (Y[k, (np.arange(J) + 2) % J]
                                                              - Y[k, (np.arange(J) - 1) % J])
        A[rows, Yidx[k, (np.arange(J) + 2) % J]] += -c * b * Y[k, (np.arange(J) + 1) % J]
        A[rows, Yidx[k, (np.arange(J) - 1) % J]] += c * b * Y[k, (np.arange(J) + 1) % J]
    return dX, dY, A


def lyap_ts(K=8, J=10, F=20.0, h=1.0, c=10.0, b=10.0, dt=0.0005,
            n_step=100000, warm=10000, seed=1, m=None):
    """状态用 RK4 推进（前向 Euler 会把快变量推爆），切向用同 dt 的 Euler + QR。"""
    n = K + K * J
    m = m or n
    rng = np.random.default_rng(seed)
    X = F * np.ones(K) + 0.01 * rng.standard_normal(K)
    Y = 0.01 * rng.standard_normal((K, J))
    Q = np.linalg.qr(rng.standard_normal((n, m)))[0]
    s = np.zeros(m)

    def rhs(X, Y):
        dX = (-np.roll(X, 1) * (np.roll(X, 2) - np.roll(X, -1)) - X + F
              - (h * c / b) * Y.sum(axis=1))
        dY = (-c * b * np.roll(Y, -1, axis=1) * (np.roll(Y, -2, axis=1) - np.roll(Y, 1, axis=1))
              - c * Y + (h * c / b) * X[:, None])
        return dX, dY

    for i in range(warm + n_step):
        if i >= warm:
            _, _, A = ts_l96_jac(X, Y, K, J, F, h, c, b)
            Q = Q + dt * (A @ Q)
            Q, R = np.linalg.qr(Q)
            s += np.log(np.abs(np.diag(R)) + 1e-300)
        k1x, k1y = rhs(X, Y)
        k2x, k2y = rhs(X + dt / 2 * k1x, Y + dt / 2 * k1y)
        k3x, k3y = rhs(X + dt / 2 * k2x, Y + dt / 2 * k2y)
        k4x, k4y = rhs(X + dt * k3x, Y + dt * k3y)
        X = X + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        Y = Y + dt / 6 * (k1y + 2 * k2y + 2 * k3y + k4y)
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
            raise RuntimeError(f"发散于第 {i} 步")
    return s / (n_step * dt)


def ky(lyap):
    ly = np.sort(lyap)[::-1]
    cum = np.cumsum(ly)
    pos = np.where(cum > 0)[0]
    if pos.size == 0:
        return 0.0, 0
    k = int(pos[-1])
    npos = int(np.sum(ly > 0))
    if k + 1 >= len(ly):
        return float(len(ly)), npos
    return float(k + 1 + cum[k] / abs(ly[k + 1])), npos


def main():
    out = {}
    print("=" * 100)
    print("双尺度 Lorenz-96 的 Lyapunov 谱（此前只算过肘部）")
    print("=" * 100)
    for K in (8, 4):
        J, c, b = 10, 10.0, 10.0
        n = K + K * J
        ly = lyap_ts(K=K, J=J, c=c, b=b, m=min(n, 60))
        dky, npos = ky(ly)
        out[f"K{K}"] = dict(K=K, dim=n, d_ky=dky, n_positive=npos,
                            lyap=[float(v) for v in np.sort(ly)[::-1]])
        print(f"  K={K} (维数 {n}): D_KY={dky:.3f}  正指数={npos}")
        print("    前 12 个指数: " + " ".join(f"{v:+.4f}" for v in np.sort(ly)[::-1][:12]))

    # 汇总：所有已知 (D_KY, d*) 对
    pairs = [
        ("KS L=22", 5.244, 8), ("KS L=33 (行波)", 2.000, 3), ("KS L=44", 9.894, 18),
        ("KS L=55", 12.411, 22), ("KS L=66", 14.811, 28), ("KS L=77", 17.320, 32),
        ("KS L=88", 19.886, 37),
    ]
    for K in (4, 8):
        d_elbow = {4: 4, 8: 8}[K]
        pairs.append((f"双尺度 L96 K={K}", out[f"K{K}"]["d_ky"], d_elbow))
    print("\n" + "=" * 100)
    print("引理检验：d* >= D_KY ?")
    print("=" * 100)
    print(f"{'系统':<20}{'D_KY':>10}{'d*':>6}{'d* - D_KY':>12}{'通过?':>8}")
    ok = 0
    for nm, dky, ds in pairs:
        good = ds >= dky
        ok += good
        print(f"{nm:<20}{dky:>10.3f}{ds:>6}{ds-dky:>12.3f}{('是' if good else '否'):>8}")
    out["lemma_check"] = dict(pairs=[(nm, dky, ds) for nm, dky, ds in pairs],
                              n_pass=int(ok), n_total=len(pairs))
    print(f"\n通过 {ok}/{len(pairs)}")
    print("注：KS L=33 是行波（D_KY=2.00），d*=3；单尺度 L96 与 2D Kolmogorov 无肘部，引理在此为空。")
    json.dump(out, open(os.path.join(HERE, "v31_lemma_check.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved v31_lemma_check.json")


if __name__ == "__main__":
    main()
