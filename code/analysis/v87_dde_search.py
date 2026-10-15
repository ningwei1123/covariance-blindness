# -*- coding: utf-8 -*-
"""V87（方向 C）：判据引导地寻找第二个 A 级系统族 —— 时滞微分方程（DDE）。

为什么选 DDE
  · **无穷维但谱离散**：线性化有有限个不稳定模，存在明确谱隙 —— 正是"奴役"最容易成立的结构；
  · 有**真正的先验子空间**：`V` = 线性化的不稳定特征模在 [−τ,0] 上张成的空间
    （由方程本身算出，**不含任何数据信息**，故判据不会退化）；
  · 时滞系统在激光、生理、控制中是标准模型；
  · 我此前 11 族筛选中**从未试过**。

模型（Mackey-Glass）
    dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)

方法
  1. 用小步长 RK4 + 历史插值积分；
  2. 状态 = x 在 [t-tau, t] 上 M 个采样点上的值；
  3. 线性化特征方程  lambda + gamma = f'(x*) exp(-lambda tau)，求 Re(lambda) > 0 的模
     ⇒ V = 这些模的实/虚部在网格上取值张成的空间（dim V = 不稳定模数）；
  4. 算判据 margin = sigma_min(P_{W_d}|V) - R_max（d 取 dim V）；
  5. 跑检测器找肘部，并用打乱尾部做对照。
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v20_robust import elbow                      # noqa: E402
from v34_audit import sig_profile, shuffle_tail   # noqa: E402
from v50_graph_slope import graph_slope           # noqa: E402

OUT = {}


def mg_rhs(x, xd, beta=0.2, gamma=0.1, n=10.0):
    return beta * xd / (1.0 + xd ** n) - gamma * x


def integrate_mg(tau=17.0, dt=0.01, T=4000.0, warm=600.0, seed=1, M=64, stride=None):
    """返回状态矩阵 X（每行 = [t-tau, t] 上 M 个采样点）。"""
    beta, gamma, n = 0.2, 0.1, 10.0
    nhist = int(round(tau / dt))
    rng = np.random.default_rng(seed)
    hist = 0.9 + 0.05 * rng.standard_normal(nhist)      # 历史函数
    x = hist[-1]

    def hist_at(s):
        """插值取 t-s 处的历史值，0 <= s <= tau。"""
        k = s / dt
        i = int(np.floor(k))
        f = k - i
        i0 = len(hist) - 1 - i
        i1 = i0 - 1
        if i1 < 0:
            return hist[max(i0, 0)]
        return (1 - f) * hist[i0] + f * hist[i1]

    snaps = []
    if stride is None:
        stride = max(1, int(round(0.05 * tau / dt)))
    nstep = int(T / dt)
    for it in range(nstep):
        xd = hist_at(tau)
        k1 = mg_rhs(x, xd, beta, gamma, n)
        k2 = mg_rhs(x + dt / 2 * k1, xd, beta, gamma, n)
        k3 = mg_rhs(x + dt / 2 * k2, xd, beta, gamma, n)
        k4 = mg_rhs(x + dt * k3, xd, beta, gamma, n)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        hist = np.append(hist[1:], x)
        if it * dt > warm and (it % stride == 0):
            th = np.linspace(-tau, 0.0, M)
            snaps.append([hist_at(-t) for t in th])
    return np.asarray(snaps)


def unstable_modes(tau, M=64, beta=0.2, gamma=0.1, n=10.0):
    """求线性化特征方程 lambda + gamma = fp * exp(-lambda tau) 的不稳定根。"""
    # 不动点
    from scipy.optimize import brentq
    f = lambda x: beta * x / (1 + x ** n) - gamma * x
    xs = brentq(f, 1e-6, 1.0)
    fp = beta * (1 + (1 - n) * xs ** n) / (1 + xs ** n) ** 2
    roots = []
    # 在复平面上粗扫 + 牛顿细化
    for re in np.linspace(-gamma - 1.0, 1.0, 60):
        for im in np.linspace(-6.0, 6.0, 120):
            lam = complex(re, im)
            for _ in range(60):
                g = lam + gamma - fp * np.exp(-lam * tau)
                dg = 1 + fp * tau * np.exp(-lam * tau)
                if abs(dg) < 1e-14:
                    break
                lam = lam - g / dg
            if abs(lam + gamma - fp * np.exp(-lam * tau)) < 1e-8 and lam.real > 0:
                if not any(abs(lam - r) < 1e-4 for r in roots):
                    roots.append(lam)
    return sorted(roots, key=lambda z: -z.real), xs, fp


def main():
    print("=" * 104)
    print("V87（方向 C）  时滞微分方程（Mackey-Glass）：判据引导的第二族搜索")
    print("=" * 104)
    M = 64
    for tau in (17.0, 23.0, 30.0, 40.0):
        roots, xs, fp = unstable_modes(tau, M=M)
        nu = len(roots)
        print(f"\n  ── tau = {tau}   不动点 x* = {xs:.4f}   f'(x*) = {fp:.4f}"
              f"   不稳定模数 = {nu}")
        if nu == 0:
            print("     无不稳定模（非混沌），跳过"); continue
        X = integrate_mg(tau=tau, M=M)
        if len(X) < 500:
            print("     样本不足，跳过"); continue
        X = X - X.mean(0)
        s = np.linalg.svd(X, compute_uv=False)
        lam_ = s ** 2 / (s ** 2).sum()
        deff = float(1 / (lam_ ** 2).sum())
        # V = 不稳定特征模在网格上的取值
        th = np.linspace(-tau, 0.0, M)
        cols = []
        for r in roots:
            v = np.exp(r * th)
            cols.append(v.real)
            if abs(r.imag) > 1e-8:
                cols.append(v.imag)
        V = np.array(cols).T
        V = np.linalg.qr(V)[0][:, :min(V.shape[1], M - 1)]
        # POD
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        d = min(V.shape[1], Vt.shape[0] - 1)
        smin = float(np.linalg.svd(Vt[:d] @ V, compute_uv=False)[-1])
        excl = max(5, int(0.05 * tau / 0.01))
        gs = graph_slope(X, V, excl=excl)
        marg = smin - gs["R_max"]
        ds = list(range(2, min(35, X.shape[1] - 1)))
        res, el = elbow(X, ds, deg=2, nval=min(2000, len(X) // 5))
        so = sig_profile({dd: res[dd]["ratio"] for dd in ds}, ds)
        de_el = el[0]
        Xs, _ = shuffle_tail(X, de_el)
        res2, _ = elbow(Xs, ds, deg=2, nval=min(2000, len(X) // 5))
        ss = sig_profile({dd: res2[dd]["ratio"] for dd in ds}, ds)
        s_o, s_s = float(so.get(de_el, np.nan)), float(ss.get(de_el, np.nan))
        print(f"     状态维 {X.shape[1]}  样本 {len(X)}  d_eff = {deff:.2f}")
        print(f"     判据: dim V = {d}  sigma_min = {smin:.4f}  R_max = {gs['R_max']:.4f}"
              f"  margin = {marg:+.4f}  {'**认证**' if marg > 0 else '拒绝'}")
        print(f"     检测器: 肘部 = {de_el}  s_orig = {s_o:.3f}  s_shuf = {s_s:.3f}"
              f"  {'耦合驱动' if (s_s > 0.9 and s_o < 0.7) else '未通过'}")
        OUT[f"tau{tau}"] = dict(tau=tau, nu=nu, dim_V=int(d), d_eff=deff,
                                sigma_min=smin, R_max=gs["R_max"], margin=marg,
                                elbow=int(de_el), s_orig=s_o, s_shuf=s_s)
    json.dump(OUT, open(os.path.join(HERE, "v87_dde_search.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
    print("\nsaved v87_dde_search.json")


if __name__ == "__main__":
    main()
