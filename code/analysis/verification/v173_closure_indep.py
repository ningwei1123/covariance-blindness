# -*- coding: utf-8 -*-
"""V173：独立复算**闭包改善**（审计中唯一未验证的头条量化结论）。

待验数字（来自 v85/v86）
  · 在 d_M 处：相对状态误差 截断 0.1757/0.1769/0.1783 → 闭包 0.0354/0.0380/0.0389
               改善 4.97× / 4.66× / 4.58×
  · 匹配精度：r(eps)=21/43/64 → 闭包同精度需 17/33/49（省 1.24× / 1.30× / 1.31×）

独立实现要点
  · 截断误差：‖X − P_d X‖_F / ‖X‖_F
  · 闭包误差：把尾部 r 用头的**二次多项式**闭式拟合（岭回归闭解），
              重建为 [z, F(z)C]，误差 ‖X − 重建‖_F / ‖X‖_F
  · 匹配精度：在闭包曲线上找最小的 d 使其误差不超过截断在 r(eps) 处的误差
  全部用留出集评估（前 80% 训练，后 20% 测试），与 v85/v86 的协议一致。
"""
import json, os, sys
import numpy as np

H = r"E:\deepseekworkspace\experiments_v12"
OUT = {}


def poly2(Z):
    n, d = Z.shape
    cols = [np.ones((n, 1)), Z]
    for i in range(d):
        for j in range(i, d):
            cols.append((Z[:, i] * Z[:, j])[:, None])
    return np.hstack(cols)


def run(X, ds, test_frac=0.2, lam=1e-9):
    n = len(X)
    ntr = int(n * (1 - test_frac))
    mu = X[:ntr].mean(0, keepdims=True)
    Xtr, Xte = X[:ntr] - mu, X[ntr:] - mu
    tot = np.linalg.norm(Xte)
    _, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    out = {}
    for d in ds:
        Phi = Vt[:d]
        ztr, zte = Xtr @ Phi.T, Xte @ Phi.T
        Rtr = Xtr - ztr @ Phi
        Ftr, Fte = poly2(ztr), poly2(zte)
        C = np.linalg.solve(Ftr.T @ Ftr + lam * np.eye(Ftr.shape[1]), Ftr.T @ Rtr)
        rec = zte @ Phi + Fte @ C
        out[d] = dict(trunc=float(np.linalg.norm(Xte - zte @ Phi) / tot),
                      close=float(np.linalg.norm(Xte - rec) / tot))
    return out


def main():
    print("=" * 100)
    print("V173  独立复算闭包改善")
    print("=" * 100)
    files = {22: ("v20_snap_L22_N64.npy", 8), 44: ("v20_snap_L44_N64.npy", 18),
             66: ("v20_snap_L66_N72.npy", 28)}
    print(f"  {'L':>4}{'d_M':>5}{'截断@d_M':>11}{'闭包@d_M':>11}{'改善':>8}"
          f"{'参考值':>9}")
    for L, (fn, dM) in files.items():
        p = os.path.join(H, fn)
        if not os.path.exists(p):
            print(f"  {L:>4}  缺 {fn}"); continue
        X = np.asarray(np.load(p, mmap_mode="r"), dtype=np.float64)
        ds = list(range(2, min(45, X.shape[1] - 1)))
        r = run(X, ds)
        tM, cM = r[dM]["trunc"], r[dM]["close"]
        ref = {22: 4.97, 44: 4.66, 66: 4.58}[L]
        print(f"  {L:>4}{dM:>5}{tM:>11.4f}{cM:>11.4f}{tM/cM:>8.2f}{ref:>9.2f}")
        # 匹配精度
        r_eps = min(d for d in ds if r[d]["trunc"] <= 1e-3)
        d_close = min(d for d in ds if r[d]["close"] <= r[r_eps]["trunc"])
        print(f"       匹配精度：r(1e-3)={r_eps}（截断 {r[r_eps]['trunc']:.4f}）"
              f" → 闭包需 d={d_close}  省 {r_eps/d_close:.2f}×")
        OUT[f"L={L}"] = dict(d_M=dM, trunc_at_dM=tM, close_at_dM=cM, improv=tM / cM,
                             r_eps=int(r_eps), d_close_same_acc=int(d_close),
                             saved=r_eps / d_close)
    json.dump(OUT, open(os.path.join(H, "v173_closure_indep.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print("\n  验收：改善应在 4.5–5.0×、匹配精度节省应在 1.2–1.35×。")
    print("saved v173_closure_indep.json")


if __name__ == "__main__":
    main()
