# -*- coding: utf-8 -*-
"""V166：最小化检查 —— KS L=66 在两种测试集大小下的肘部（deg=2，快）。"""
import json, os, sys
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import v20_robust

OUT = {}
logs = []


def P(s):
    print(s, flush=True)
    logs.append(s)


P("=" * 88)
P("V166  KS 肘部 vs 测试集大小（deg=2）")
P("=" * 88)
P(f"  {'L':>4}{'n':>7}{'nval':>7}{'elbow':>7}{'w@elbow':>11}{'w@dM':>11}{'dM':>5}")
dM = {22: 8, 44: 18, 66: 28}
for L, fn in ((22, "v20_snap_L22_N64.npy"), (44, "v20_snap_L44_N64.npy"),
              (66, "v20_snap_L66_N72.npy")):
    p = os.path.join(H, fn)
    if not os.path.exists(p):
        P(f"  {L}: 缺 {fn}"); continue
    X0 = np.asarray(np.load(p, mmap_mode="r"), dtype=np.float64)
    M = len(X0)
    for n, nval in ((4000, 500), (4000, 2000), (M, 500), (M, 2000)):
        if nval >= n - 500 or n > M:
            continue
        X = X0[:: max(1, M // n)][:n]
        ds = list(range(2, min(36, X.shape[1] - 1)))
        try:
            res, el = v20_robust.elbow(X, ds, deg=2, nval=nval)
        except Exception as e:
            P(f"  {L:>4}{len(X):>7}{nval:>7}   失败 {type(e).__name__}")
            continue
        we = res.get(el[0], {}).get("w", float("nan"))
        wm = res.get(dM[L], {}).get("w", float("nan"))
        P(f"  {L:>4}{len(X):>7}{nval:>7}{el[0]:>7}{we:>11.4f}{wm:>11.4f}{dM[L]:>5}")
        OUT.setdefault(f"L={L}", []).append(
            dict(n=len(X), nval=nval, elbow=int(el[0]), w_elbow=float(we),
                 w_dM=float(wm)))

json.dump(OUT, open(os.path.join(H, "v166_sensitivity.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=str)
open(os.path.join(H, "v166_sensitivity.txt"), "w", encoding="utf-8").write("\n".join(logs))
P("\n  已存 v166_sensitivity.json / .txt")
