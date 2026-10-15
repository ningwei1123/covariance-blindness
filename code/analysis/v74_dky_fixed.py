# -*- coding: utf-8 -*-
"""V74：修正 D_KY —— 剔除守恒均值模带来的虚假零指数。

发现的错误（由文献严格比对暴露）
  v20_ks_core.KSSystem 的切向束 dU 活在 N=64 维**实**空间里，QR 也在这空间做。
  但 KS 的 k=0 均值模**严格守恒**（_N(v) 在 k=0 处恒为 0），所以切向的常数分量不变，
  **必然产生一个恒为 0 的 Lyapunov 指数**。它是守恒律，不是动力学中性模。

  后果：D_KY = j + (Σ_{i≤j}λ_i)/|λ_{j+1}| 中的 j 被抬高 1 ⇒ **D_KY 全部多算 1**。

  证据（L=22）
      我的谱  : 0.0484, 0.0003, 0.0002, -0.0002, -0.0032, -0.1869, ...
      文献谱  : 0.048,      0,      0,          -0.003,  -0.189,  ...   (Cvitanović+ 2010)
      我多出 -0.0002 这一个；去掉后 D_KY = 4 + 0.0457/0.1869 = 4.2445
      文献             D_KY = 4 + 0.045 /0.189  = 4.238          **吻合**

修法：QR 之前把切向束的均值减去，即把它投影到**零均值子空间**（63 维）。
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from v20_ks_core import etdrk4_coeffs, wavenumbers, KSSystem   # noqa: E402

OUT = {}


class KSFixed(KSSystem):
    """与 KSSystem 相同，但切向束每次 QR 前投影到零均值子空间。"""

    def step(self):
        v, dv = self.v, self.dU
        E, E2, Q, f1, f2, f3 = self.E, self.E2, self.Q, self.f1, self.f2, self.f3
        Qc, f1c, f2c, f3c = Q[:, None], f1[:, None], f2[:, None], f3[:, None]
        Nv = self._N(v)
        Ndv = self._DN(v, dv)
        a = E2 * v + Q * Nv
        da = E2[:, None] * dv + Qc * Ndv
        Na, Nda = self._N(a), self._DN(a, da)
        b = E2 * v + Q * Na
        db = E2[:, None] * dv + Qc * Nda
        Nb, Ndb = self._N(b), self._DN(b, db)
        c = E2 * a + Q * (2.0 * Nb - Nv)
        dc = E2[:, None] * da + Qc * (2.0 * Ndb - Ndv)
        Nc, Ndc = self._N(c), self._DN(c, dc)
        self.v = E * v + f1 * Nv + 2.0 * f2 * (Na + Nb) + f3 * Nc
        dv_new = (E[:, None] * dv + f1c * Ndv + 2.0 * f2c * (Nda + Ndb) + f3c * Ndc)
        dU_new = np.fft.irfft(dv_new, self.N, axis=0)
        dU_new = dU_new - dU_new.mean(axis=0, keepdims=True)   # <<< 剔除守恒均值方向
        Qr, R = np.linalg.qr(dU_new)
        dg = np.abs(np.diag(R))
        dg = np.where(dg > 0, dg, 1e-300)
        self.lyap_sum += np.log(dg)
        self.dU = np.fft.rfft(Qr, axis=0)
        self.t += self.dt
        self.n_steps += 1


def lyap_fixed(L, N, dt=0.1, T=4000.0, nlyap=40, warm=1000.0, seed=11):
    s = KSFixed(N=N, L=L, dt=dt, nlyap=nlyap, seed=seed)
    for _ in range(int(round(warm / dt))):
        s.step()
    s.lyap_sum[:] = 0.0
    s.t = 0.0
    for _ in range(int(round(T / dt))):
        s.step()
    return np.sort(s.lyapunov())[::-1]


def dky_of(lam):
    c = np.cumsum(lam)
    pos = np.where(c >= -1e-9)[0]
    if pos.size == 0 or pos[-1] + 1 >= len(lam):
        # 无正指数（非混沌：不动点/行波/准周期）——KY 公式不适用，返回中性模数
        neut = int(np.sum(lam >= -1e-9))
        return float(neut), neut, 0.0, float("nan")
    j = int(pos[-1])
    return float(j + 1 + c[j] / abs(lam[j + 1])), j + 1, float(c[j]), float(abs(lam[j + 1]))


def main():
    print("=" * 104)
    print("V74  修正后的 Lyapunov 谱与 D_KY（剔除守恒均值模的虚假零指数）")
    print("=" * 104)
    cases = [("L22_N64", 22.0, 64), ("L33_N64", 33.0, 64), ("L44_N64", 44.0, 64),
             ("L55_N96", 55.0, 96), ("L66_N72", 66.0, 72),
             ("L77_N128", 77.0, 128), ("L88_N160", 88.0, 160)]
    print(f"  {'域':<10}{'λ1':>9}{'λ2':>9}{'λ3':>9}{'λ4':>9}{'λ5':>9}"
          f"{'D_KY(新)':>11}{'D_KY(旧)':>11}{'差':>6}")
    old = {"L22_N64": 5.2435, "L33_N64": 2.0, "L44_N64": 9.8939, "L55_N96": 12.411,
           "L66_N72": 14.8115, "L77_N128": 17.320, "L88_N160": 19.886}
    for tag, L, N in cases:
        t0 = time.time()
        lam = lyap_fixed(L, N)
        dky_new, j, cs, nxt = dky_of(lam)
        o = old.get(tag)
        OUT[tag] = dict(L=L, N=N, lam=[float(x) for x in lam[:12]],
                        d_ky=dky_new, j=j, cum=c_ if (c_ := float(cs)) else cs,
                        next_lam=nxt, d_ky_old=o)
        print(f"  {tag:<10}" + "".join(f"{x:>9.4f}" for x in lam[:5]) +
              f"{dky_new:>11.4f}{(o if o else float('nan')):>11.4f}"
              f"{(dky_new - o if o else float('nan')):>6.1f}"
              f"   ({time.time()-t0:.0f}s)", flush=True)
    json.dump(OUT, open(os.path.join(HERE, "v74_dky_fixed.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n  [literature] Cvitanovic, Davidchack & Siminos 2010, L=22:")
    print("    8 physical modes: (0.048, 0, 0, -0.003, -0.189, -0.256, -0.290, -0.310)")
    print("    -> D_KY = 4 + 0.045/0.189 = 4.238")
    l22 = OUT["L22_N64"]
    print(f"    corrected L22 first 8 = {[round(x,4) for x in l22['lam'][:8]]}")
    print(f"    D_KY = {l22['d_ky']:.4f}   dev vs literature "
          f"{abs(l22['d_ky']-4.238)/4.238*100:.2f}%")
    json.dump(OUT, open(os.path.join(HERE, "v74_dky_fixed.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nsaved v74_dky_fixed.json")


if __name__ == "__main__":
    main()
