# -*- coding: utf-8 -*-
"""V178：宇称律的**正确陈述**与检验。

前两轮的陈述错误
  V176/V177 把"degree-q 特征"理解为**所有 ≤ q 次**单项式，于是低次项
  （如 z_0 与 z_0³ 之间 E = 3 ≠ 0）仍能解释方差，使"不可见"的判据失效。

正确陈述（本脚本检验）
  设 r 为**齐次 p 次**、零均值的多项式，特征集为**恰好 k 次**的单项式集合。
  对奇数阶矩为零的测度（高斯）：E[m_k · m_p] 的总次数为 k+p，
  故非零的必要条件是 **k + p 为偶数**。
  因此：恰好 k 次特征对纯 p 次图**不可见 ⟺ p + k 为奇数**。

再检验更强的版本（"次数过滤"）：特征集取"恰好 1 次 ∪ 恰好 2 次 ∪ … ∪ 恰好 q 次"
中与 p 同奇偶的那些次数，应可解释大部分方差；异奇偶的次数应解释 0。
"""
import json, os
from itertools import combinations_with_replacement
import numpy as np

H = r"E:\deepseekworkspace\experiments_v12"
OUT = {}


def homo(z, k, cap=60):
    """恰好 k 次的单项式集合（列数封顶，保持可算）。"""
    if k == 0:
        return np.ones((z.shape[0], 1))
    idx = list(combinations_with_replacement(range(z.shape[1]), k))
    if len(idx) > cap:
        idx = idx[:cap]
    M = np.ones((z.shape[0], len(idx)))
    for j, ii in enumerate(idx):
        for i in ii:
            M[:, j] *= z[:, i]
    return M


def explained(F, r):
    F = F - F.mean(0)
    C, *_ = np.linalg.lstsq(F, r, rcond=None)
    return 1 - ((r - F @ C) ** 2).sum() / max((r ** 2).sum(), 1e-30)


def main():
    print("=" * 100)
    print("V178  宇称律（恰好 k 次特征 × 纯 p 次图）")
    print("=" * 100)
    rng = np.random.default_rng(11)
    n, d = 40000, 6

    print(f"\n  {'p':>3}{'k':>3}{'p+k':>5}{'解释比例':>11}  {'预期':<10}判读")
    ok = tot = 0
    for p in (1, 2, 3, 4):
        for k in (1, 2, 3, 4):
            z = rng.standard_normal((n, d))
            Mp = homo(z, p); Mp = Mp - Mp.mean(0)          # 纯 p 次、零均值
            c = rng.standard_normal(Mp.shape[1])
            r = (Mp @ c)[:, None]
            Mk = homo(z, k)
            frac = explained(Mk, r)
            invisible = (p + k) % 2 == 1
            good = (frac < 0.02) if invisible else (frac > 0.10)
            ok += int(good); tot += 1
            print(f"  {p:>3}{k:>3}{p+k:>5}{frac:>11.4f}  "
                  f"{'不可见' if invisible else '可见':<10}{'✓' if good else '**✗**'}")
    print(f"\n  宇称律符合 {ok}/{tot}")

    # ---- 联合特征集（与 p 同奇偶的所有次数）
    print("\n  [联合] 特征集取与 p 同奇偶的次数，应可解释；取异奇偶应解释 0")
    for p in (2, 3):
        z = rng.standard_normal((n, d))
        Mp = homo(z, p); Mp = Mp - Mp.mean(0)
        c = rng.standard_normal(Mp.shape[1])
        r = (Mp @ c)[:, None]
        same = np.hstack([homo(z, k) for k in range(1, 5) if (k + p) % 2 == 0])
        opp = np.hstack([homo(z, k) for k in range(1, 5) if (k + p) % 2 == 1])
        print(f"    p={p}: 同奇偶特征解释 {explained(same, r):.4f}；"
              f"异奇偶特征解释 {explained(opp, r):.4f}")
        OUT[f"joint_p{p}"] = dict(same=float(explained(same, r)),
                                  opp=float(explained(opp, r)))

    OUT["parity_exact_k"] = dict(ok=ok, tot=tot)
    json.dump(OUT, open(os.path.join(H, "v178_parity.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print("\n  判读：恰好 k 次特征对纯 p 次图不可见 ⟺ p+k 为奇数。")
    print("saved v178_parity.json")


if __name__ == "__main__":
    main()
