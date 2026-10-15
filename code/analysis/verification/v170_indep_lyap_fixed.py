# -*- coding: utf-8 -*-
"""V170：独立重算 KS 的 Lyapunov 谱（修正 V169 的两处错误）。

V169 的错误
  KS:  u_t = −u u_x − u_xx − u_xxxx
  线性化: δ_t = −(u_x δ + u δ_x) + (k² − k⁴) δ
  V169 写成了 (k⁴ − k²)（符号反）且**漏掉 u_x δ 项**，故 λ1 = 95、Σλ = +3629。

本版
  状态与切向都用**同一组 ETDRK4 系数**推进；切向的中间级用状态在该级的取值。
  切向在去均值子空间上做 QR（均值模严格守恒）。
"""
import json, os
import numpy as np

H = r"E:\deepseekworkspace\experiments_v12"
OUT = {}


def etdrk4(lam, dt, M=32):
    LR = dt * lam[:, None] + np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)[None, :]
    E = np.exp(dt * lam)
    E2 = np.exp(dt * lam / 2)
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1))
    return E, E2, Q, f1, f2, f3


def ks_lyap(L, N, dt=0.05, T=2000.0, warm=400.0, seed=1):
    k = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)
    lam = k ** 2 - k ** 4                          # 正确的线性算子
    E, E2, Q, f1, f2, f3 = etdrk4(lam, dt)
    nk = len(k)
    rng = np.random.default_rng(seed)
    u = 0.1 * np.cos(2 * np.pi * np.arange(N) / L) + 1e-2 * rng.standard_normal(N)
    v = np.fft.rfft(u)

    def Nl(vv):
        uu = np.fft.irfft(vv, N)
        return -0.5j * k * np.fft.rfft(uu ** 2)

    def Ls(coef, Y):                                # 在 Fourier 域乘以系数
        return np.fft.irfft(coef[:, None] * np.fft.rfft(Y, axis=0), n=N, axis=0)

    def Nt(xs, xs_x, Y):                            # 切向非线性：−(x' δ + x δ')
        Yh = np.fft.rfft(Y, axis=0)
        Yx = np.fft.irfft(1j * k[:, None] * Yh, n=N, axis=0)
        return -(xs_x[:, None] * Y + xs[:, None] * Yx)

    for _ in range(int(warm / dt)):
        Nv = Nl(v); a = E2 * v + Q * Nv; Na = Nl(a)
        b = E2 * v + Q * Na; Nb = Nl(b)
        c = E2 * a + Q * (2 * Nb - Nv); Nc = Nl(c)
        v = E * v + f1 * Nv + 2 * f2 * (Na + Nb) + f3 * Nc

    P = np.eye(N) - np.ones((N, N)) / N
    Qf = np.linalg.qr(P @ rng.standard_normal((N, N)))[0]
    s = np.zeros(N)
    nst = int(T / dt)
    for _ in range(nst):
        x = np.fft.irfft(v, N)
        x_x = np.fft.irfft(1j * k * v, n=N)
        # 状态各级
        Nv = Nl(v)
        a = E2 * v + Q * Nv;               xa = np.fft.irfft(a, N)
        ax = np.fft.irfft(1j * k * a, n=N)
        Na = Nl(a)
        b = E2 * v + Q * Na;               xb = np.fft.irfft(b, N)
        bx = np.fft.irfft(1j * k * b, n=N)
        Nb = Nl(b)
        c = E2 * a + Q * (2 * Nb - Nv);    xc = np.fft.irfft(c, N)
        cx = np.fft.irfft(1j * k * c, n=N)
        Nc = Nl(c)
        v = E * v + f1 * Nv + 2 * f2 * (Na + Nb) + f3 * Nc
        # 切向各级（同一组系数）
        Nv_t = Nt(x, x_x, Qf)
        a_t = Ls(E2, Qf) + Ls(Q, Nv_t);        Na_t = Nt(xa, ax, a_t)
        b_t = Ls(E2, Qf) + Ls(Q, Na_t);        Nb_t = Nt(xb, bx, b_t)
        c_t = Ls(E2, a_t) + Ls(Q, 2 * Nb_t - Nv_t); Nc_t = Nt(xc, cx, c_t)
        Qf = (Ls(E, Qf) + Ls(f1, Nv_t) + 2 * Ls(f2, Na_t + Nb_t) + Ls(f3, Nc_t))
        Qf = P @ Qf
        Qf, R = np.linalg.qr(Qf)
        dg = np.abs(np.diag(R)); dg = np.where(dg > 0, dg, 1e-300)
        s += np.log(dg)
    ly = np.sort(s / (nst * dt))[::-1]
    return ly


def main():
    print("=" * 104)
    print("V170  独立重算 KS Lyapunov 谱（修正后）")
    print("=" * 104)
    lit = {22: (8, 0.048), 44: (18, 0.083), 66: (28, 0.087)}
    print(f"  {'L':>4}{'N':>5}{'λ1':>10}{'λ1(文献)':>10}{'偏差%':>8}"
          f"{'Σλ':>11}{'解析 −L':>10}{'偏差%':>8}{'D_KY':>8}{'d_M':>5}")
    for L, N in ((22, 64), (44, 64), (66, 72)):
        try:
            ly = ks_lyap(L, N, T=1500.0, warm=300.0)
        except Exception as e:
            print(f"  {L:>4}{N:>5}   失败 {type(e).__name__}: {e}")
            continue
        l1 = float(ly[0]); sl = float(ly.sum())
        cs = np.cumsum(ly); j = int(np.sum(cs >= 0))
        dky = (j + cs[j - 1] / abs(ly[j])) if 0 < j < len(ly) else float(j)
        dM, lit1 = lit[L]
        print(f"  {L:>4}{N:>5}{l1:>10.4f}{lit1:>10.4f}{(l1-lit1)/lit1*100:>8.1f}"
              f"{sl:>11.3f}{-L:>10.1f}{abs(sl+L)/L*100:>8.2f}{dky:>8.3f}{dM:>5}")
        OUT[f"L={L}"] = dict(lam1=l1, lam1_lit=lit1, sum_lam=sl, d_ky=float(dky),
                             d_M=dM, rel_sum=abs(sl + L) / L)
    json.dump(OUT, open(os.path.join(H, "v170_indep_lyap.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print("\n  验收：Σλ 对解析散度 −L 的偏差应 <2%；D_KY(22) 应 ≈ 4.2。")
    print("saved v170_indep_lyap.json")


if __name__ == "__main__":
    main()
