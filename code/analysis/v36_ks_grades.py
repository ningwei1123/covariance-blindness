# -*- coding: utf-8 -*-
"""V36：把 KS 族剩余域长的**打乱尾部检验**补齐，完成 A/B/C/D 分级。

V34 只对 L=22、L=44 跑了 P3（打乱尾部），其余五个域长是"A 级候选"。
本脚本对 L=33 / L=55 / L=66 / L=77 补跑 P3，并给出完整分级表。
（L=88 的 d*=37，二次特征数 703、快照 20000，代价可控，一并跑。）
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v34_audit import audit, verdict      # noqa: E402
from v31_lemma_check import ky            # noqa: E402

DKY = {"L22_N64": 5.244, "L33_N64": 2.000, "L44_N64": 9.894,
       "L55_N96": 12.411, "L66_N192": 14.811, "L77_N128": 17.320, "L88_N160": 19.886}
PLAN = [("L33_N64", 3, list(range(2, 13))),
        ("L55_N96", 22, list(range(16, 31))),
        ("L66_N192", 28, list(range(22, 37))),
        ("L77_N128", 32, list(range(26, 41))),
        ("L88_N160", 37, list(range(30, 44)))]

if __name__ == "__main__":
    out = {}
    print("=" * 112)
    print("KS 族：补齐打乱尾部检验（P3）")
    print("=" * 112)
    print(f"{'设定':<12}{'d*':>5}{'D_KY':>8}{'P1':>5}{'γ(肘部)':>9}{'s_orig':>9}"
          f"{'s_shuf':>9}{'P3':>10}  等级")
    for tag, d_h, ds in PLAN:
        f = os.path.join(HERE, f"v20_snap_{tag}.npy")
        if not os.path.exists(f):
            print(f"{tag:<12} 缺数据"); continue
        X = np.load(f).astype(np.float64)
        t0 = time.time()
        r = audit(tag, X, d_h, ds)
        d_e, dky = r["d_elbow"], DKY[tag]
        p1 = "过" if d_e >= dky else "违反"
        if r["s_shuf"] > 0.9:
            p3, grade = "耦合驱动", "A"
        else:
            p3, grade = "断崖掩蔽", "B"
        if p1 == "违反":
            grade = "C(撤回)"
        r.update(d_ky=dky, P1=p1, P3=p3, grade=grade)
        out[tag] = r
        print(f"{tag:<12}{d_e:>5}{dky:>8.2f}{p1:>5}{r['gamma_at_elbow']:>9.2f}"
              f"{r['s_orig']:>9.3f}{r['s_shuf']:>9.3f}{p3:>10}  {grade}"
              f"   ({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(os.path.join(HERE, "v36_ks_grades.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    print("\n===== KS 族最终分级 =====")
    for k, v in out.items():
        print(f"  {k:<12} d*={v['d_elbow']:<3} D_KY={v['d_ky']:.2f}  γ={v['gamma_at_elbow']:.2f}  "
              f"s_orig={v['s_orig']:.3f}  s_shuf={v['s_shuf']:.3f}  -> {v['grade']}")
    print("\nsaved v36_ks_grades.json")
