# -*- coding: utf-8 -*-
"""V86（方向 B 补完）：按**同等精度**比较，图闭合到底省多少模态？

为什么必须补
  稿件现有表述「标准判据把流形维数高估 2.3 倍」隐含"方差判据浪费了 2.3 倍模态"。
  但 V85 显示：`r(1e-3) = 20` 是达到 1e-3 精度所需，而 `d_M = 8` 处的截断误差是 0.176。
  两者在回答**不同问题**。必须按**同等精度**重算节省量。
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v85_rom_sizing import errors_vs_d, pod_basis   # noqa: E402

OUT = {}


def main():
    print("=" * 100)
    print("V86  按同等精度比较：图闭合 vs 截断")
    print("=" * 100)
    print(f"  {'KS':>8}{'d_M':>5}{'r(1e-3)':>9}{'截断@r':>12}{'闭合所需 d':>12}"
          f"{'节省':>8}{'比值':>8}{'改进@d_M':>10}")
    for tag, dM in (("L22_N64", 8), ("L44_N64", 18), ("L66_N72", 28)):
        f = os.path.join(HERE, f"v20_snap_{tag}.npy")
        if not os.path.exists(f):
            continue
        X = np.load(f).astype(np.float64)
        n = len(X); ntr = int(0.7 * n)
        Xtr, Xte = X[:ntr], X[ntr:]
        Xc = Xtr - Xtr.mean(0)
        s = np.linalg.svd(Xc, compute_uv=False)
        e = np.cumsum(s[::-1] ** 2)[::-1]
        r_eps = int(np.searchsorted(-e, -1e-6 * e[0]) + 1)
        ds = list(range(2, X.shape[1] - 1))
        rows = errors_vs_d(Xtr, Xte, ds)
        t_at_r = next(t for d, t, c, m in rows if d == r_eps)
        # 图闭合达到同一精度所需的最小 d
        need_c = next((d for d, t, c, m in rows if c <= t_at_r), ds[-1])
        t_at_dM = next(t for d, t, c, m in rows if d == dM)
        c_at_dM = next(c for d, t, c, m in rows if d == dM)
        print(f"  {tag:>8}{dM:>5}{r_eps:>9}{t_at_r:>12.3e}{need_c:>12}"
              f"{r_eps - need_c:>8}{r_eps / need_c:>8.2f}{t_at_dM / c_at_dM:>10.1f}x")
        OUT[tag] = dict(d_M=dM, r_eps=r_eps, trunc_at_r=t_at_r,
                        d_close_same_acc=need_c,
                        saved_modes=r_eps - need_c, ratio=r_eps / need_c,
                        improv_at_dM=t_at_dM / c_at_dM)
    print("\n  判读：")
    print("    · `改进@d_M` 是**强而一致**的结果：在 d* 处加闭式闭合，状态误差降约 5 倍。")
    print("    - 比值才是「按同等精度省多少模态」，远小于 2.3，故稿件原表述需改。")
    json.dump(OUT, open(os.path.join(HERE, "v86_rom_matched.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nsaved v86_rom_matched.json")


if __name__ == "__main__":
    main()
