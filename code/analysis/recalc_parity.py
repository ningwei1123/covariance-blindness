# -*- coding: utf-8 -*-
"""用「恰好 k 次」特征集重算完整 4x4 宇称律网格（与论文命题一致）。"""
import io, os
from itertools import combinations_with_replacement
import numpy as np

D = r"E:\投稿文章\英文\PRL\仓库文件\data"
rng = np.random.default_rng(11)
n, d = 40000, 6


def homo(z, k, cap=60):
    """恰好 k 次的单项式集合。"""
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
    return float(1 - ((r - F @ C) ** 2).sum() / max((r ** 2).sum(), 1e-30))


rows, ok = [], 0
for p in (1, 2, 3, 4):
    for q in (1, 2, 3, 4):
        z = rng.standard_normal((n, d))
        Mp = homo(z, p)
        Mp = Mp - Mp.mean(0)
        c = rng.standard_normal(Mp.shape[1])
        r = (Mp @ c)[:, None]
        frac = explained(homo(z, q), r)
        invisible = (p + q) % 2 == 1
        good = (frac < 0.02) if invisible else (frac > 0.10)
        ok += int(good)
        rows.append([p, q, f"{frac:.6f}", "invisible" if invisible else "visible",
                     "yes" if good else "NO"])
        print(f"  p={p} q={q}  p+q={p+q}  explained={frac:.4f}  "
              f"{'invisible' if invisible else 'visible':<10} {'OK' if good else '**不符**'}")

with io.open(os.path.join(D, "fig2c_parity_grid.txt"), "w", encoding="utf-8",
             newline="\n") as fh:
    fh.write("# Fig. 2(c). Parity law with EXACTLY-k-degree feature sets. A pure "
             "degree-p graph is invisible to exactly-q-degree features if and only if "
             "p+q is odd. All 16 combinations tested.\n")
    fh.write("# graph_degree_p\tfeature_degree_q\texplained_fraction\tprediction\tagrees\n")
    for r in rows:
        fh.write("\t".join(str(v) for v in r) + "\n")
print(f"\n  宇称律 {ok}/16 符合 -> fig2c_parity_grid.txt")
