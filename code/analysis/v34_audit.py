# -*- coding: utf-8 -*-
"""V34：谱断崖保护规则 + 对**所有**系统（含我自己的合成对照）重审。

背景（V33）：双尺度 L96 的肘部被证明是 POD 谱断崖造成的假阳性，
因为它在「打乱尾部」对照下**逐点不变**；而 KS 的肘部被打乱尾部**摧毁**。

现在必须回答一个更尖锐的问题：
**我的合成"已知真值"对照（d0 = 5/8/12，7/8 命中）会不会也是谱断崖假阳性？**
若会，则检测器目前唯一经过验证的正面案例只剩 KS 本身。

判据（每条都要看）：
  γ(d) = sv(d)/sv(d+1)                       谱断崖指标
  s(d) = ratio(d) / median(同奇偶近邻 ratio)   肘部显著性（越小越孤立）
  s_shuf(d) = 同一量在「打乱尾部」数据上的值
  判定：
    真奴役  : s(d_elbow) < 0.7 且 s_shuf(d_elbow) > 0.9（肘部被打乱摧毁）且 γ(d_elbow) < 3
    谱断崖  : s_shuf(d_elbow) ≈ s(d_elbow)（打乱后不变）
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v20_robust import elbow                       # noqa: E402
from v20_injectivity import make_aligned_graph     # noqa: E402
from v27_twoscale import ts_l96                    # noqa: E402


def shuffle_tail(X, d_head, seed=0):
    """保留经验协方差（POD 基下交叉协方差本就为 0），只破坏头-尾函数关系。"""
    Xc = X - X.mean(0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt.T
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Z))
    Z2 = Z.copy()
    Z2[:, d_head:] = Z[perm][:, d_head:]
    return Z2 @ Vt, s


def sig_profile(ratio, ds):
    out = {}
    for d in ds:
        if np.isnan(ratio.get(d, np.nan)):
            continue
        nb = [ratio[k] for k in ds if k != d and abs(k - d) <= 6
              and (k - d) % 2 == 0 and not np.isnan(ratio.get(k, np.nan))]
        if len(nb) >= 3:
            out[d] = ratio[d] / float(np.median(nb))
    return out


def audit(tag, X, d_head, ds):
    res_o, el_o = elbow(X, ds, deg=2, nval=min(2000, len(X) // 4))
    Xs, s = shuffle_tail(X, d_head)
    res_s, el_s = elbow(Xs, ds, deg=2, nval=min(2000, len(X) // 4))
    ro = {d: res_o[d]["ratio"] for d in ds}
    rs = {d: res_s[d]["ratio"] for d in ds}
    so, ss = sig_profile(ro, ds), sig_profile(rs, ds)
    sv = s / s[0]
    gam = {d: float(sv[d - 1] / sv[d]) for d in ds if d < len(sv)}
    d_e = el_o[0]
    return dict(tag=tag, d_elbow=d_e,
                s_orig=float(so.get(d_e, np.nan)), s_shuf=float(ss.get(d_e, np.nan)),
                ratio_orig=float(ro.get(d_e, np.nan)), ratio_shuf=float(rs.get(d_e, np.nan)),
                gamma_at_elbow=float(gam.get(d_e, np.nan)),
                gamma_max=float(max(gam.values())) if gam else float("nan"),
                d_elbow_shuf=el_s[0], so={str(k): v for k, v in so.items()},
                ss={str(k): v for k, v in ss.items()},
                gamma={str(k): v for k, v in gam.items()}, ds=list(ds))


def verdict(r):
    if not np.isfinite(r["s_orig"]):
        return "无显著肘部"
    if r["ratio_orig"] >= r["ratio_shuf"] * 0.75 and abs(r["s_orig"] - r["s_shuf"]) < 0.15:
        return "谱断崖假阳性"
    if r["gamma_at_elbow"] > 3.0:
        return "疑谱断崖（保护规则触发）"
    if r["s_orig"] < 0.7 and r["s_shuf"] > 0.9:
        return "真奴役"
    if r["s_orig"] < r["s_shuf"] - 0.15:
        return "弱奴役信号"
    return "不确定"


def main():
    cases = []
    # 1) KS
    for tag, d_h, ds in (("L22_N64", 8, list(range(3, 16))),
                         ("L44_N64", 18, list(range(12, 27)))):
        X = np.load(os.path.join(HERE, f"v20_snap_{tag}.npy")).astype(np.float64)
        cases.append((f"KS {tag}", X, d_h, ds))
    # 2) 合成对齐流形（我自己的"已知真值"对照）
    for d0, amp in ((5, 0.02), (8, 0.02), (12, 0.005)):
        X, _ = make_aligned_graph(d0, amp=amp, seed=20 + d0)
        cases.append((f"合成对齐 d0={d0} amp={amp}", X, d0, list(range(2, d0 + 8))))
    # 3) 双尺度 L96（已知假阳性）
    Xt, _ = ts_l96(K=8, J=10, c=10.0, b=10.0, dt=0.0005, T=400.0, stride=20, seed=3)
    cases.append(("双尺度 L96 K=8", Xt, 8, list(range(3, 16))))
    # 4) 单尺度 L96（已知无肘部）
    Xl = np.load(os.path.join(HERE, "v21_snap_L96_N20.npy")).astype(np.float64)
    cases.append(("单尺度 L96 N=20", Xl, 10, list(range(4, 20))))

    out = {}
    print("=" * 118)
    print("谱断崖保护规则 + 打乱尾部对照：对所有系统重审")
    print("=" * 118)
    print(f"{'系统':<26}{'肘部':>5}{'s(orig)':>9}{'s(shuf)':>9}"
          f"{'ratio原':>9}{'ratio乱':>9}{'γ(肘部)':>9}{'γ最大':>8}  判定")
    for tag, X, d_h, ds in cases:
        r = audit(tag, X, d_h, ds)
        v = verdict(r)
        r["verdict"] = v
        out[tag] = r
        print(f"{tag:<26}{r['d_elbow']:>5}{r['s_orig']:>9.3f}{r['s_shuf']:>9.3f}"
              f"{r['ratio_orig']:>9.3f}{r['ratio_shuf']:>9.3f}"
              f"{r['gamma_at_elbow']:>9.2f}{r['gamma_max']:>8.2f}  {v}", flush=True)

    json.dump(out, open(os.path.join(HERE, "v34_audit.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n===== 汇总 =====")
    for k, v in out.items():
        print(f"  {k:<26} {v['verdict']}")
    print("saved v34_audit.json")


if __name__ == "__main__":
    main()
