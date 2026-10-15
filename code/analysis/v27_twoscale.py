# -*- coding: utf-8 -*-
"""V27：第二个系统族的**正面复现** + 机理的受控检验 —— 双尺度 Lorenz-96。

    dX_k/dt = -X_{k-1}(X_{k-2} - X_{k+1}) - X_k + F - (h c / b) * sum_j Y_{j,k}
    dY_{j,k}/dt = -c b Y_{j+1,k}(Y_{j+2,k} - Y_{j-1,k}) - c Y_{j,k} + (h c / b) X_k

  K=8 慢变量，每个慢变量配 J=10 个快变量 -> 全状态维数 88。
  c = 时间尺度分离比。c=1 时几乎无分离（接近单尺度 L96），c=10~20 时快变量被强烈阻尼。

**受控预测（事先写下）**：
  · c=1   : 无尺度分离 -> 应**无肘部**（与单尺度 L96 的负面结论一致）；
  · c=10/20: 快变量被奴役 -> 全 88 维状态应是**慢变量上的图** -> 应出现**孤立肘部**，
             且位置远小于 88（预计落在个位数到十几）；
  · 伴随检验：k-近邻 G(d) 在同样位置下降；快变量相对慢变量的条件方差塌缩。
  若肘部随 c 单调开启，则「肘部 = 奴役开维数」得到受控确认，并同时完成第二系统族复现。
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v20_robust import elbow            # noqa: E402
from v22c_robust import g_curve         # noqa: E402


def ts_l96(K=8, J=10, F=20.0, h=1.0, c=10.0, b=10.0, dt=0.001, T=1000.0,
           stride=50, warm=200.0, seed=1):
    rng = np.random.default_rng(seed)
    X = F * np.ones(K) + 0.01 * rng.standard_normal(K)
    Y = 0.01 * rng.standard_normal((K, J))

    def rhs(X, Y):
        dX = (-np.roll(X, 1) * (np.roll(X, 2) - np.roll(X, -1)) - X + F
              - (h * c / b) * Y.sum(axis=1))
        dY = (-c * b * np.roll(Y, -1, axis=1) * (np.roll(Y, -2, axis=1) - np.roll(Y, 1, axis=1))
              - c * Y + (h * c / b) * X[:, None])
        return dX, dY

    def step(X, Y):
        k1x, k1y = rhs(X, Y)
        k2x, k2y = rhs(X + dt / 2 * k1x, Y + dt / 2 * k1y)
        k3x, k3y = rhs(X + dt / 2 * k2x, Y + dt / 2 * k2y)
        k4x, k4y = rhs(X + dt * k3x, Y + dt * k3y)
        return (X + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x),
                Y + dt / 6 * (k1y + 2 * k2y + 2 * k3y + k4y))

    for _ in range(int(warm / dt)):
        X, Y = step(X, Y)
    snaps, slow = [], []
    for i in range(int(T / dt)):
        X, Y = step(X, Y)
        if i % stride == 0:
            snaps.append(np.concatenate([X, Y.ravel()]))
            slow.append(X.copy())
    return np.asarray(snaps), np.asarray(slow)


def cond_var_ratio(slow, full):
    """快变量在给定慢变量后的残余方差占比（用 k-近邻条件均值估计）。

    slow: (T, K)   full: (T, 88)
    返回快变量部分 (I - E[.|slow]) 的相对残差。
    """
    T = len(slow)
    tr = np.arange(0, T - 2000)
    te = np.arange(T - 2000, T)
    rng = np.random.default_rng(0)
    tr = rng.choice(tr, size=min(6000, len(tr)), replace=False)
    full = full - full.mean(0)
    Ftr, Fte = full[tr], full[te]
    Str, Ste = slow[tr], slow[te]
    k = 15
    pred = np.empty_like(Fte)
    for a0 in range(0, len(te), 200):
        a1 = min(a0 + 200, len(te))
        D = ((Ste[a0:a1] ** 2).sum(1)[:, None] + (Str ** 2).sum(1)[None, :]
             - 2 * Ste[a0:a1] @ Str.T)
        idx = np.argpartition(D, k, axis=1)[:, :k]
        pred[a0:a1] = Ftr[idx].mean(axis=1)
    return float(np.linalg.norm(Fte - pred) / np.linalg.norm(Fte - Ftr.mean(0)))


def main():
    out = {}
    print("=" * 104)
    print("双尺度 Lorenz-96：K=8, J=10 -> 88 维；扫描尺度分离比 c")
    print("=" * 104)
    print(f"{'c':>5}{'dt':>8}{'快照':>8}{'肘部':>6}{'比值':>8}{'最显著d':>8}{'s':>8}"
          f"{'G最小':>8}{'@d':>4}{'快变量条件残差':>16}")
    for c in (1.0, 2.0, 5.0, 10.0, 20.0):
        dt = 0.001 if c <= 10 else 0.0005
        T = 1000.0 if c <= 10 else 500.0
        stride = 50 if c <= 10 else 25
        t0 = time.time()
        X, slow = ts_l96(c=c, dt=dt, T=T, stride=stride, seed=3)
        N = X.shape[1]
        ds = list(range(2, min(N - 1, 30)))
        res, el = elbow(X, ds, deg=2, nval=2000)
        # 显著性
        ds_s = [d for d in ds if not np.isnan(res[d]["ratio"])]
        st = {}
        for d in ds_s:
            nb = [res[k]["ratio"] for k in ds_s if k != d and abs(k - d) <= 6
                  and (k - d) % 2 == 0]
            if len(nb) >= 3:
                st[d] = res[d]["ratio"] / float(np.median(nb))
        d_sig = min(st, key=st.get) if st else None
        G = g_curve(X, ds[:0] or list(range(2, 20)), n_train=min(9000, len(X) // 2),
                    n_query=500)
        gm = min(G, key=G.get)
        cvr = cond_var_ratio(slow, X)
        out[f"c{c:g}"] = dict(c=c, dt=dt, n=len(X), dim=N, elbow_d=el[0],
                              argmin_stat=float(el[1]), d_sig=d_sig,
                              sig_stat=float(st[d_sig]) if d_sig else None,
                              G_min=float(G[gm]), G_min_d=gm,
                              G={str(k): v for k, v in G.items()},
                              cond_var_ratio=cvr,
                              ratio=[res[d]["ratio"] for d in ds])
        print(f"{c:>5.1f}{dt:>8.4f}{len(X):>8}{str(el[0]):>6}{el[1]:>8.3f}"
              f"{str(d_sig):>8}{(st[d_sig] if d_sig else float('nan')):>8.3f}"
              f"{G[gm]:>8.3f}{gm:>4}{cvr:>16.3f}   ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(os.path.join(HERE, "v27_twoscale.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    print("\nsaved v27_twoscale.json")


if __name__ == "__main__":
    main()
