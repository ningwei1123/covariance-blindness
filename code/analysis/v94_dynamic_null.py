# -*- coding: utf-8 -*-
"""V94（算法层对照）：补上**动态零假设** —— 相位随机化替代数据。

为什么必须补
  我们的尾部打乱是 Theiler 框架下的**静态零假设**（保持瞬时二阶块结构）。
  标准替代数据（Theiler 1992；Schreiber & Schmitz 1996 的 IAAFT）是**动态零假设**：
  保持**功率谱**（时间线性结构）而随机化相位。
  两者检验的是**不同的**零假设，因此必须两个都报，否则"肘部不是谱假象"这一论断
  就缺少它对应的对照。

做法
  对每个 POD 系数时间序列做相位随机化（保持幅度谱 -> 保持自相关/功率谱），
  重组后跑检测器与显著性统计。
  预期：肘部被破坏（若肘部只是功率谱的假象，则相位随机化后仍会出现 -> 那才是问题）。
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v20_robust import elbow                     # noqa: E402
from v34_audit import sig_profile, shuffle_tail  # noqa: E402

OUT = {}


def phase_randomize(X, seed=0):
    """逐列相位随机化：保持每列的幅度谱（= 功率谱），随机化相位。"""
    rng = np.random.default_rng(seed)
    n, m = X.shape
    Xc = X - X.mean(0)
    F = np.fft.rfft(Xc, axis=0)
    ph = np.exp(2j * np.pi * rng.random(F.shape))
    ph[0] = 1.0                       # 直流不动
    if n % 2 == 0:
        ph[-1] = 1.0                  # Nyquist 不动
    Y = np.fft.irfft(np.abs(F) * ph, n=n, axis=0)
    return Y


def main():
    print("=" * 100)
    print("V94  动态零假设（相位随机化）作为补充对照")
    print("=" * 100)
    for tag, dM in (("L22_N64", 8), ("L44_N64", 18)):
        f = os.path.join(HERE, f"v20_snap_{tag}.npy")
        if not os.path.exists(f):
            continue
        X = np.load(f).astype(np.float64)
        if len(X) > 8000:
            X = X[::max(1, len(X) // 8000)]
        ds = list(range(3, 25))
        res, el = elbow(X, ds, deg=2, nval=800)
        so = sig_profile({d: res[d]["ratio"] for d in ds}, ds)
        de = el[0]
        s_o = float(so.get(de, np.nan))
        # 静态零假设（我们的打乱）
        Xs, _ = shuffle_tail(X, de, seed=0)
        r2, _ = elbow(Xs, ds, deg=2, nval=800)
        s2 = sig_profile({d: r2[d]["ratio"] for d in ds}, ds)
        s_shuf = float(s2.get(de, np.nan))
        el_shuf = None
        # 动态零假设（相位随机化），多种子
        els, ss = [], []
        for sd in range(4):
            Y = phase_randomize(X, seed=sd)
            r3, e3 = elbow(Y, ds, deg=2, nval=800)
            s3 = sig_profile({d: r3[d]["ratio"] for d in ds}, ds)
            els.append(int(e3[0]))
            ss.append(float(s3.get(e3[0], np.nan)))
        # 相位随机化数据在**原肘部**处的显著性（是否仍有孤立下降）
        s_at_de = []
        for sd in range(4):
            Y = phase_randomize(X, seed=sd)
            r4, _ = elbow(Y, ds, deg=2, nval=800)
            s4 = sig_profile({d: r4[d]["ratio"] for d in ds}, ds)
            s_at_de.append(float(s4.get(de, np.nan)))
        print(f"\n  ── KS {tag}（真值 d_M = {dM}）")
        print(f"     原始       : 肘部 = {de}   s = {s_o:.3f}")
        print(f"     静态零假设 : 肘部 = {de}   s_shuf = {s_shuf:.3f}"
              f"   -> {'肘部被破坏 ✓' if s_shuf > 0.9 else '未破坏'}")
        print(f"     动态零假设 : 各次肘部 = {els}   在 d={de} 处 s = "
              f"{[round(v,2) for v in s_at_de]}")
        print(f"                  （相位随机化后肘部位置应**不稳定**或消失，"
              f"且原肘部处 s 应失去孤立性）")
        OUT[tag] = dict(d_M=dM, elbow=int(de), s_orig=s_o, s_shuf_static=s_shuf,
                        elbow_phase=els, s_phase_at_de=s_at_de)
    json.dump(OUT, open(os.path.join(HERE, "v94_dynamic_null.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nsaved v94_dynamic_null.json")


if __name__ == "__main__":
    main()
