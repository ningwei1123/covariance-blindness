# -*- coding: utf-8 -*-
"""重渲染 PRL 三张图，使其与正文图注的面板一一对应。

Fig. 1  (a)(b)(c)(d)  协方差盲性
Fig. 2  (a)(b)(c)(d)  机理：四阶矩三条路径 / 解析值 / 宇称律 / 尾部打乱
Fig. 3  (a)(b)(c)(d)  反例（方差阈值高估 2.3--2.6 倍）/ 判据 margin / 肘点 vs 域长 /
                      Mackey--Glass 判据与肘点 vs 延迟
"""
import io, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.backends.backend_pdf import PdfPages

H = r"E:\deepseekworkspace\experiments_v12"
D = r"E:\投稿文章\英文\PRL\仓库文件\data"
OUT = r"E:\deepseekworkspace\PRL_submission"
J = lambda f: json.load(open(os.path.join(H, f), encoding="utf-8"))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.2,
    "axes.titlesize": 8.5, "legend.fontsize": 6.8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "axes.grid": False, "legend.frameon": False, "figure.dpi": 600,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.025,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
MM = 1 / 25.4
W2 = 178 * MM
BL, OR, GR, RD, PU = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"


def txt(name):
    rows = []
    for ln in io.open(os.path.join(D, name), encoding="utf-8"):
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            rows.append(ln.split("\t"))
    return rows


def panel(ax, s, dx=-0.16, dy=1.05):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            va="bottom", ha="left")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print(f"    [OK] {name}.png / .pdf")


# ============================================================== FIG 1
def fig1():
    r = txt("fig1_covariance_blindness.txt")
    names = [x[0] for x in r]
    dG = np.array([float(x[1]) for x in r])
    dM = np.array([float(x[2]) for x in r])
    cov = np.array([float(x[3]) for x in r])
    uG = np.array([float(x[4]) for x in r])
    uM = np.array([float(x[5]) for x in r])
    x = np.arange(len(names))
    lbl = [n.replace("_", "\n") for n in names]

    fig, ax = plt.subplots(2, 2, figsize=(W2, 118 * MM), constrained_layout=True)

    ax[0, 0].bar(x - 0.2, dG, 0.4, color=BL, label="Gaussian $\\mu_G$")
    ax[0, 0].bar(x + 0.2, dM, 0.4, color=RD, label="graph $\\mu_M$")
    ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels(lbl, fontsize=6)
    ax[0, 0].set_ylabel("participation ratio $d_{\\rm eff}$")
    ax[0, 0].legend(loc="upper left")
    panel(ax[0, 0], "(a)")

    ax[0, 1].semilogy(x, cov, "o", ms=6, color=GR)
    ax[0, 1].axhline(2.22e-16, color="k", ls="--", lw=0.9, label="machine $\\epsilon$")
    ax[0, 1].set_xticks(x); ax[0, 1].set_xticklabels(lbl, fontsize=6)
    ax[0, 1].set_ylabel("$\\|C_G-C_M\\|/\\|C\\|$")
    ax[0, 1].legend(loc="lower right")
    panel(ax[0, 1], "(b)")

    ax[1, 0].bar(x - 0.2, uG, 0.4, color=BL, label="Gaussian")
    ax[1, 0].bar(x + 0.2, uM, 0.4, color=RD, label="graph")
    ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(lbl, fontsize=6)
    ax[1, 0].set_ylabel("unexplained tail variance")
    ax[1, 0].set_ylim(0, 1.1)
    ax[1, 0].legend(loc="center right")
    panel(ax[1, 0], "(c)")

    sep = uG / np.maximum(uM, 1e-12)
    cols = [BL if "lin" in n else OR for n in names]
    ax[1, 1].bar(x, sep, color=cols)
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(lbl, fontsize=6)
    ax[1, 1].set_ylabel("separation ratio $u_G/u_M$")
    h = [plt.Rectangle((0, 0), 1, 1, color=BL), plt.Rectangle((0, 0), 1, 1, color=OR)]
    ax[1, 1].legend(h, ["linear spectrum", "geometric spectrum"], loc="lower right")
    panel(ax[1, 1], "(d)")

    fig.savefig(os.path.join(OUT, "PRL_Fig1.png"), dpi=300)
    with PdfPages(os.path.join(OUT, "PRL_Fig1.pdf")) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print("    [OK] PRL_Fig1.png")


# ============================================================== FIG 2
def fig2():
    ra = txt("fig2a_fourth_moment_routes.txt")
    rb = txt("fig2b_fourth_moment_analytic.txt")
    rg = txt("fig2c_parity_grid.txt")
    rs = txt("fig2a_elbow_significance.txt")

    fig, ax = plt.subplots(2, 2, figsize=(W2, 118 * MM), constrained_layout=True)

    # (a) 三路径
    ds = [x[0] for x in ra]
    p12 = [float(x[1]) for x in ra]
    p13 = [float(x[2]) for x in ra]
    xx = np.arange(len(ds))
    ax[0, 0].semilogy(xx - 0.12, p12, "o", ms=7, color=BL,
                      label="normal eq. vs SVD pinv")
    ax[0, 0].semilogy(xx + 0.12, p13, "s", ms=6, color=RD,
                      label="normal eq. vs residual")
    ax[0, 0].axhline(2.22e-16, color="k", ls="--", lw=0.9, label="machine $\\epsilon$")
    ax[0, 0].set_xticks(xx); ax[0, 0].set_xticklabels([f"$d$={v}" for v in ds])
    ax[0, 0].set_ylabel("max relative difference")
    ax[0, 0].set_ylim(1e-16, 1e-13)
    ax[0, 0].legend(loc="lower left")
    panel(ax[0, 0], "(a)")

    # (b) 解析对比
    ds2 = [x[0] for x in rb]
    an = [float(x[1]) for x in rb]
    nu = [float(x[2]) for x in rb]
    rel = [float(x[3]) for x in rb]
    xx2 = np.arange(len(ds2))
    ax[0, 1].bar(xx2 - 0.2, an, 0.4, color=OR, label="analytic $\\|c\\|_G^2$")
    ax[0, 1].bar(xx2 + 0.2, nu, 0.4, color=BL, label="closed form")
    for i, r_ in enumerate(rel):
        ax[0, 1].annotate(f"{r_:.0e}", (i, max(an[i], nu[i])), textcoords="offset points",
                          xytext=(0, 3), ha="center", fontsize=6.2)
    ax[0, 1].set_xticks(xx2)
    ax[0, 1].set_xticklabels([f"$d$={v}" for v in ds2])
    ax[0, 1].set_ylabel("$\\Delta_k$")
    ax[0, 1].set_ylim(0, max(max(an), max(nu)) * 1.22)
    ax[0, 1].legend(loc="lower right")
    panel(ax[0, 1], "(b)")

    # (c) 宇称律 4x4
    M = np.full((4, 4), np.nan)
    for x_ in rg:
        p, q, frac = int(x_[0]), int(x_[1]), float(x_[2])
        if 1 <= p <= 4 and 1 <= q <= 4:
            M[p - 1, q - 1] = frac
    im = ax[1, 0].imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1, origin="lower")
    for p in range(4):
        for q in range(4):
            v = M[p, q]
            if np.isnan(v):
                continue
            ax[1, 0].text(q, p, f"{v:.2f}", ha="center", va="center", fontsize=6.4,
                          color="white" if v > 0.55 or v < 0.08 else "black")
    for p in range(4):
        for q in range(4):
            if (p + 1 + q + 1) % 2 == 1:
                ax[1, 0].add_patch(plt.Rectangle((q - .5, p - .5), 1, 1, fill=False,
                                                 ec="k", lw=1.4, ls="--"))
    ax[1, 0].set_xticks(range(4)); ax[1, 0].set_xticklabels(range(1, 5))
    ax[1, 0].set_yticks(range(4)); ax[1, 0].set_yticklabels(range(1, 5))
    ax[1, 0].set_xlabel("feature degree $q$")
    ax[1, 0].set_ylabel("graph degree $p$")
    ax[1, 0].set_title("dashed: $p+q$ odd $\\rightarrow$ invisible", fontsize=7, pad=3)
    plt.colorbar(im, ax=ax[1, 0], label="explained fraction", pad=0.02)
    panel(ax[1, 0], "(c)")

    # (d) tail shuffle 七域
    Ls = [int(float(x[0])) for x in rs]
    so = [float(x[2]) for x in rs]
    ss = [float(x[3]) for x in rs]
    xx3 = np.arange(len(Ls))
    ax[1, 1].bar(xx3 - 0.2, so, 0.4, color=BL, label="original")
    ax[1, 1].bar(xx3 + 0.2, ss, 0.4, color=RD, label="tail shuffled")
    ax[1, 1].axhline(0.9, color="k", ls="--", lw=0.8)
    ax[1, 1].set_xticks(xx3)
    ax[1, 1].set_xticklabels([f"{v}" for v in Ls], fontsize=7)
    ax[1, 1].set_xlabel("domain size $L$")
    ax[1, 1].set_ylabel("elbow significance $s$")
    ax[1, 1].legend(loc="upper right")
    panel(ax[1, 1], "(d)")

    fig.savefig(os.path.join(OUT, "PRL_Fig2.png"), dpi=300)
    with PdfPages(os.path.join(OUT, "PRL_Fig2.pdf")) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print("    [OK] PRL_Fig2.png")


# ============================================================== FIG 3
def fig3():
    ra = txt("fig3a_matched_accuracy.txt")
    rb = txt("fig3b_ekman_friction.txt")
    rc = txt("fig3b_criterion_all.txt")
    rs = txt("fig2a_elbow_significance.txt")
    rb2 = txt("fig2b_dimension_vs_domain.txt")
    rmg = txt("fig2c_mackey_glass.txt")
    rmgr = txt("mackey_glass_robustness.txt")

    fig, ax = plt.subplots(2, 2, figsize=(W2, 118 * MM), constrained_layout=True)

    # (a) 反例：方差阈值 vs 已验证维数
    Ls = [int(float(x[0])) for x in ra]
    dM = [int(float(x[1])) for x in ra]
    r_eps = [int(float(x[2])) for x in ra]
    xx = np.arange(len(Ls))
    ax[0, 0].bar(xx - 0.2, r_eps, 0.4, color=OR,
                 label="threshold rank $r(10^{-3})$")
    ax[0, 0].bar(xx + 0.2, dM, 0.4, color=BL, label="verified dimension $d_M$")
    for i in range(len(Ls)):
        ax[0, 0].annotate(f"{r_eps[i]/dM[i]:.2f}$\\times$", (i, r_eps[i]),
                          textcoords="offset points", xytext=(0, 3), ha="center",
                          fontsize=7.5, fontweight="bold")
    ax[0, 0].set_xticks(xx); ax[0, 0].set_xticklabels([f"$L$={v}" for v in Ls])
    ax[0, 0].set_ylabel("modes")
    ax[0, 0].set_ylim(0, max(r_eps) * 1.22)
    ax[0, 0].legend(loc="upper left")
    panel(ax[0, 0], "(a)")

    # (b) 判据 margin vs d_eff（含 certified）
    # 列序：system, d_eff, d, sigma_min, R_max, margin, verdict
    h = [plt.Line2D([], [], marker="o", ls="", color=BL),
         plt.Line2D([], [], marker="s", ls="", color=RD)]
    for x_ in rc:
        nm = x_[0]
        de = float(x_[1])
        mg = float(x_[5])                      # margin, not sigma_min
        pos = mg > 0
        ax[0, 1].scatter(de, mg, s=48, color=BL if pos else RD,
                         marker="o" if pos else "s", zorder=3,
                         edgecolors="none")
        short = (nm.replace("single-scale ", "").replace("two-scale ", "2-scale ")
                 .replace("Mackey-Glass tau=", "MG $\\tau$=")
                 .replace("2D Kolmogorov d=", "2D Kolm. $d$="))
        if nm.startswith("KS"):
            ax[0, 1].annotate(short, (de, mg), textcoords="offset points",
                              xytext=(0, -11), fontsize=6, ha="center")
        elif nm.startswith("2D") or "scale L96" in nm:
            ax[0, 1].annotate(short[:16], (de, mg), textcoords="offset points",
                              xytext=(6, 2), fontsize=6)
    ax[0, 1].axhline(0, color="k", lw=1.1)
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_xlim(1.4, 90)
    ax[0, 1].set_xticks([2, 5, 10, 20, 50])
    ax[0, 1].set_xticklabels(["2", "5", "10", "20", "50"])
    ax[0, 1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0, 1].set_ylim(-18.5, 2.2)
    ax[0, 1].set_xlabel("participation ratio $d_{\\rm eff}$")
    ax[0, 1].set_ylabel("criterion margin")
    ax[0, 1].legend(h, ["certified", "rejected"], loc="center left")
    panel(ax[0, 1], "(b)")

    # (c) 肘点 vs 域长 + 文献
    Le = [int(float(x[0])) for x in rs]
    el = [int(float(x[1])) for x in rs]
    lit = {int(float(x[0])): int(float(x[2])) for x in rb2 if x[2] != "NA"}
    lk = sorted(lit)
    ax[1, 0].plot(Le, el, "o-", ms=5, color=BL, label="estimator (7 domains)")
    ax[1, 0].plot(lk, [lit[k] for k in lk], "s--", ms=6, color=RD,
                  label="inertial manifold (lit.)")
    for k in lk:
        ax[1, 0].annotate(f"{lit[k]}", (k, lit[k]), textcoords="offset points",
                          xytext=(5, -8), fontsize=6.5, color=RD)
    for L_, e_ in zip(Le, el):
        if L_ == 33:
            ax[1, 0].annotate("traveling\nwave", (L_, e_), textcoords="offset points",
                              xytext=(2, 6), fontsize=6, color="k", ha="left")
    ax[1, 0].set_xlabel("domain size $L$")
    ax[1, 0].set_ylabel("dimension $d^{*}$")
    ax[1, 0].legend(loc="upper left")
    panel(ax[1, 0], "(c)")

    # (d) Mackey-Glass：判据 margin 与肘点 vs 延迟
    taus = [float(x[0]) for x in rmg]
    elb = [int(float(x[2])) for x in rmg]
    byl = {}
    for x_ in rmgr:
        t_ = float(x_[0]); m_ = float(x_[4])
        byl.setdefault(t_, []).append(m_)
    tt = sorted(byl)
    ax[1, 1].axhline(0, color="k", lw=1.0)
    for i, t_ in enumerate(tt):
        v = byl[t_]
        ax[1, 1].scatter([t_] * len(v), v, s=26, color=BL if min(v) > 0 else RD,
                         zorder=3, alpha=0.85)
    ax[1, 1].plot(tt, [np.median(byl[t_]) for t_ in tt], "k-", lw=1.0, zorder=2)
    ax[1, 1].set_xlabel("delay $\\tau$")
    ax[1, 1].set_ylabel("criterion margin", color="k")
    ax2 = ax[1, 1].twinx()
    ax2.plot(taus, elb, "D--", ms=5, color=GR, label="elbow")
    ax2.set_ylabel("elbow", color=GR)
    ax2.tick_params(axis="y", colors=GR)
    h = [plt.Line2D([], [], marker="o", ls="", color=BL),
         plt.Line2D([], [], marker="o", ls="", color=RD),
         plt.Line2D([], [], marker="D", ls="--", color=GR)]
    ax[1, 1].legend(h, ["certified", "rejected", "elbow"], loc="lower right")
    panel(ax[1, 1], "(d)")

    fig.savefig(os.path.join(OUT, "PRL_Fig3.png"), dpi=300)
    with PdfPages(os.path.join(OUT, "PRL_Fig3.pdf")) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print("    [OK] PRL_Fig3.png")


if __name__ == "__main__":
    print("重渲染 PRL 三图（每图 4 面板，与图注严格一致）...")
    fig1(); fig2(); fig3()
    print("完成")
