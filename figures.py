"""Every figure in the manuscript, drawn from the stored results in data/.

This file never runs the model. Run `python model.py --sweeps` first to
produce data/, then `python figures.py` to draw from it.

Usage
-----
    python figures.py              all figures
    python figures.py 2 3 6        only those

Figures
-------
    1  pca_modes             MU-modes identified by PCA with varimax rotation
    2  fig2_bars_both        mode firing frequency per phase window, both rates
    3  fig3_scatter_both     mode co-variation within a window, both rates
    4  fig4_cellwise         unit-by-unit correlations, both rates
    5  fig5_sweep            the 2-D inhibition-gain sweep
    6  fig6_ucm              UCM decomposition, both rates
    7  fig7_rates            indices against task rate
    8  force_level           indices against force level
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

LF, LS = '1 Hz task (1 s cycle)', 'Slow task (10 s cycle)'
CF, CS = '#c0392b', '#2471a3'
RATES = [('fast', LF, CF), ('slow', LS, CS)]
UNITS = [f'MU{i+1}' for i in range(5)]

# Near-black navy at low rates through brown to bright yellow at high rates.
# Lightness rises monotonically so the panels stay readable in greyscale and
# with colour-vision deficiency.
RATE_CMAP = LinearSegmentedColormap.from_list('navy_brown_yellow', [
    (0.00, '#060d2a'), (0.22, '#262a5c'), (0.48, '#6b4430'),
    (0.74, '#c4791c'), (1.00, '#ffd84d')])


def load(name):
    """Read one stored result, with a clear message if model.py has not run."""
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        flag = '' if name.startswith('cases') else ' --sweeps'
        sys.exit(f'{path} is missing. Run `python model.py{flag}` first.')
    return np.load(path) if name.endswith('.npz') else json.load(open(path))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote figures/{name}')


# ==========================================================================
def fig1_pca():
    """MU-modes identified by PCA, whole cycle and each half-cycle."""
    d, s = load('cases.npz'), load('cases.json')['fast']
    panels = [(d['fast_pca_loadings'],
               f'Whole cycle, {s["pca_pct"]:.1f}% of variance'),
              (d['fast_pca_loadings_up'],
               f'Force increase (recruitment), {s["pca_pct_up"]:.1f}%'),
              (d['fast_pca_loadings_down'],
               f'Force decrease (derecruitment), {s["pca_pct_down"]:.1f}%')]
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.6))
    xs = np.arange(5)
    w = 0.38
    for ax, (L, title) in zip(axs, panels):
        ax.bar(xs - w/2, L[:, 0], w, color=CF, label='MU-mode 1')
        ax.bar(xs + w/2, L[:, 1], w, color=CS, label='MU-mode 2')
        for y in (0.5, -0.5):
            ax.axhline(y, color='k', ls=':', lw=1.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(UNITS)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel('varimax loading')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=.35)
    c = s['pca_half_corr']
    fig.suptitle('MU-modes identified by principal component analysis with varimax '
                 'rotation (dotted: loading = $\\pm$0.5 acceptance criterion; '
                 f'half-cycle loading vectors correlate at {c[0]:.3f} and {c[1]:.3f})',
                 fontsize=12)
    fig.tight_layout()
    save(fig, 'pca_modes.png')


def fig2_bars():
    """Mode firing frequency across phase windows, both rates."""
    d, S = load('cases.npz'), load('cases.json')
    nb = S['fast']['n_bins']
    xb = np.arange(nb)
    target = np.sin(2*np.pi*(xb + 0.5)/nb - np.pi/2)
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.8))
    for ax, (tag, lbl, _) in zip(axs, RATES):
        bm, g1 = d[f'{tag}_bin_mean'], d[f'{tag}_g1']
        w = 0.4
        ax.bar(xb - w/2, bm[:, 0], w, color=CF, alpha=.85,
               label='MU-mode 1 (MU1-3)')
        ax.bar(xb + w/2, bm[:, 1], w, color=CS, alpha=.85,
               label='MU-mode 2 (MU4-5)')
        ax.set_xlabel('phase window (20 per cycle)')
        ax.set_ylabel('mode firing frequency [Hz]')
        ax.set_xticks(xb[::2])
        ax.grid(axis='y', ls='--', alpha=.5)
        axt = ax.twinx()
        axt.plot(xb, target, 'k--', lw=2)
        axt.set_ylabel('force target')
        ax.set_title(f'{lbl}\nr with target = {g1[0]:+.2f} / {g1[1]:+.2f}')
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    axs[0].legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    save(fig, 'fig2_bars_both.png')


def fig3_scatter(example=9):
    """One example phase window at each rate, plus r in all twenty."""
    d, S = load('cases.npz'), load('cases.json')
    nb = S['fast']['n_bins']
    xb = np.arange(nb)
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.8))
    for ax, (tag, lbl, _) in zip(axs, RATES):
        b, rr = d[f'{tag}_binned'], d[f'{tag}_r']
        x, y = b[:, example, 0], b[:, example, 1]
        ax.scatter(x, y, s=42, color='teal', alpha=.8, edgecolor='k', linewidth=.5)
        slope, icpt = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 10)
        ax.plot(xs, slope*xs + icpt, 'r-', lw=2)
        ax.set_xlabel('MU-mode 1 [Hz]')
        ax.set_ylabel('MU-mode 2 [Hz]')
        ax.set_title(f'{lbl}\nphase window {example+1}:  '
                     f'b = {slope:+.2f},  r = {rr[example]:+.2f}')
        ax.grid(alpha=.35)
    for tag, lbl, col in RATES:
        axs[2].plot(xb + 1, d[f'{tag}_r'], '-o', color=col, lw=2, ms=4, label=lbl)
    axs[2].axhline(0, color='k', lw=1)
    axs[2].set_ylim(-1, 0.3)
    axs[2].set_xlabel('phase window')
    axs[2].set_ylabel('r (mode 1 vs mode 2)')
    axs[2].set_title('All twenty windows')
    axs[2].grid(alpha=.35)
    axs[2].legend(fontsize=8)
    fig.suptitle('MU-mode co-variation across cycles, within a fixed phase window',
                 fontsize=13)
    fig.tight_layout()
    save(fig, 'fig3_scatter_both.png')


def fig4_cellwise():
    """Unit-by-unit correlations within phase windows, both rates."""
    d, S = load('cases.npz'), load('cases.json')
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for ax, (tag, lbl, _) in zip(axs, RATES):
        ub = d[f'{tag}_unit_binned']
        nb, N = S[tag]['n_bins'], 5
        C = np.mean([np.corrcoef(ub[:, j, :].T) for j in range(nb)], axis=0)
        im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1)
        for a in range(N):
            for b in range(N):
                ax.text(b, a, f'{C[a,b]:+.2f}', ha='center', va='center',
                        fontsize=10, color='w' if abs(C[a, b]) > 0.55 else 'k')
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticklabels(UNITS); ax.set_yticklabels(UNITS)
        for (x0, y0, w, h) in [(-.5, -.5, 3, 3), (2.5, 2.5, 2, 2)]:
            ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False,
                                       edgecolor='k', lw=3))
        wi = np.mean([C[a, b] for a in range(N) for b in range(N)
                      if a != b and (a < 3) == (b < 3)])
        bt = np.mean([C[a, b] for a in range(N) for b in range(N)
                      if (a < 3) != (b < 3)])
        ax.set_title(f'{lbl}\nwithin-mode r = {wi:+.2f}   '
                     f'between-mode r = {bt:+.2f}')
        plt.colorbar(im, ax=ax, label='mean within-window r')
    fig.suptitle('Motor units co-vary positively within a mode and negatively '
                 'across modes', fontsize=13, y=1.03)
    fig.tight_layout()
    save(fig, 'fig4_cellwise.png')


def fig5_sweep():
    """The 2-D inhibition-gain sweep, both Golgi scenarios."""
    d = load('sweep_gains.npz')
    WR, WIB = d['WR'], d['WIB']
    gtos = ['total', 'per_mode']
    titles = {'total': 'Golgi senses TOTAL pool force',
              'per_mode': 'Golgi senses ITS OWN mode only'}
    star = (0.5, 3.0)                       # w_Ib, wR_cross as used
    r_thr, mn_lo, mn_hi = -0.5, 8.0, 20.0
    X, Y = np.meshgrid(WIB, WR)
    rmax = max(d[f'R_{g}'].max() for g in gtos)
    rmin = min(d[f'R_{g}'].min() for g in gtos)

    fig, axs = plt.subplots(2, 2, figsize=(14.5, 11))
    for c, gm in enumerate(gtos):
        R, MN = d[f'R_{gm}'], d[f'MN_{gm}']
        ax = axs[0][c]
        if rmax > 0.05:
            im = ax.pcolormesh(X, Y, R, cmap='RdBu_r', shading='gouraud',
                               norm=TwoSlopeNorm(vmin=min(-1, rmin), vcenter=0,
                                                 vmax=max(0.2, rmax)))
        else:
            im = ax.pcolormesh(X, Y, R, cmap='Blues_r', shading='gouraud',
                               vmin=-1.0, vmax=0.0)
        cs = ax.contour(X, Y, R, levels=[-0.8, -0.6, -0.4, -0.2, 0.0],
                        colors='k', linewidths=1.2)
        ax.clabel(cs, fmt='%.1f', fontsize=9)
        mask = ((R <= r_thr) & (MN >= mn_lo) & (MN <= mn_hi)).astype(float)
        if mask.any():
            ax.contour(X, Y, mask, levels=[0.5], colors='#e67e22', linewidths=3.5)
            ax.contourf(X, Y, mask, levels=[0.5, 1.5], colors=['#e67e22'],
                        alpha=.16)
        ax.plot(*star, marker='*', ms=22, color='gold', mec='k', mew=1.3, zorder=6)
        ax.set_xlabel('Ib (Golgi) inhibition   w$_{Ib}$')
        ax.set_ylabel('Renshaw between-mode inhibition   w$_{R,cross}$')
        ax.set_title(f'{titles[gm]}\nmean within-window r (MU-mode 1 vs 2)')
        plt.colorbar(im, ax=ax, label='mean within-window r')

        ax2 = axs[1][c]
        im2 = ax2.pcolormesh(X, Y, MN, cmap=RATE_CMAP, shading='gouraud',
                             vmin=4, vmax=24)
        cb = ax2.contour(X, Y, MN, levels=[mn_lo, mn_hi], colors='#ffffff',
                         linewidths=2.5, linestyles='--')
        ax2.clabel(cb, fmt='%.0f Hz', fontsize=9)
        ax2.plot(*star, marker='*', ms=22, color='gold', mec='k', mew=1.3, zorder=6)
        ax2.set_xlabel('Ib (Golgi) inhibition   w$_{Ib}$')
        ax2.set_ylabel('Renshaw between-mode inhibition   w$_{R,cross}$')
        ax2.set_title('mean MU firing rate [Hz]\n'
                      f'dashed = {mn_lo:.0f} and {mn_hi:.0f} Hz '
                      'physiological bounds')
        plt.colorbar(im2, ax=ax2, label='MU firing rate [Hz]')
        ok = (R <= r_thr) & (MN >= mn_lo) & (MN <= mn_hi)
        print(f'  {gm:9s}: r from {R.min():+.2f} to {R.max():+.2f}, '
              f'{ok.sum()}/{ok.size} grid points usable')
    fig.suptitle('Orange: strong MU-mode co-variation (r $\\leq$ %.1f) at a '
                 'physiological firing rate (%.0f-%.0f Hz).   '
                 'star = model as used' % (r_thr, mn_lo, mn_hi), fontsize=13)
    fig.tight_layout()
    save(fig, 'fig5_sweep.png')


def fig6_ucm():
    """UCM decomposition across the cycle, both rates."""
    d, S = load('cases.npz'), load('cases.json')
    xb = np.arange(S['fast']['n_bins'])
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for tag, lbl, col in RATES:
        axs[0].plot(xb + 1, d[f'{tag}_Vucm'], '-o', color=col, lw=2, ms=4,
                    label=f'{lbl}: V$_{{UCM}}$')
        axs[0].plot(xb + 1, d[f'{tag}_Vort'], '--s', color=col, lw=2, ms=4,
                    alpha=.65, label=f'{lbl}: V$_{{ORT}}$')
        axs[1].plot(xb + 1, d[f'{tag}_dV'], '-o', color=col, lw=2, ms=4, label=lbl)
    axs[0].set_yscale('log')
    axs[0].set_ylabel('variance per DOF [Hz$^2$]')
    axs[0].set_title('Variance within the UCM and within ORT')
    axs[1].axhline(0, color='k', lw=1)
    axs[1].set_ylim(-2.1, 2.1)
    axs[1].set_ylabel('$\\Delta$V')
    axs[1].set_title('Synergy index $\\Delta$V (bounded $\\pm$2)')
    for a in axs:
        a.set_xlabel('phase window')
        a.grid(alpha=.4)
        a.legend(fontsize=8)
    fig.suptitle('UCM decomposition of inter-cycle variance in MU-mode space',
                 fontsize=13)
    fig.tight_layout()
    save(fig, 'fig6_ucm.png')


def fig7_rates():
    """Co-variation indices as a function of task rate."""
    a = np.array(load('rates.json'), float)
    cyc, r, dV, r_sd, dV_sd = a[:, 0], a[:, 1], a[:, 3], a[:, 4], a[:, 6]
    x = np.arange(len(cyc))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    axs[0].errorbar(x, r, yerr=r_sd, fmt='-o', color=CF, lw=2, ms=6, capsize=4)
    axs[0].set_ylabel('r (mode 1 vs mode 2, within window)')
    axs[0].set_ylim(-1.0, 0.05)
    axs[1].errorbar(x, dV, yerr=dV_sd, fmt='-o', color='#27ae60', lw=2, ms=6,
                    capsize=4)
    axs[1].set_ylabel('$\\Delta$V')
    axs[1].set_ylim(-0.05, 2.1)
    for ax in axs:
        ax.axhline(0, color='k', lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{c:g}' for c in cyc])
        ax.set_xlabel('cycle duration [s]')
        ax.grid(alpha=.35)
    fig.suptitle('Mode co-variation weakens as the task slows '
                 '(mean $\\pm$ SD across seeds)', fontsize=13)
    fig.tight_layout()
    save(fig, 'fig7_rates.png')


def fig8_force_level():
    """Force variability and the synergy index against force level."""
    c = load('force_sweep.json')
    d = {name: np.array([row[i] for row in c['data']])
         for i, name in enumerate(c['cols'])}
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.6))
    panels = [('cvpct', CF, 'CV of total force [%]',
               'Relative force variability rises at low force'),
              ('sdF', CS, 'SD of total force [a.u.]',
               'Absolute variability is nearly flat'),
              ('dV', '#27ae60', '$\\Delta$V',
               'Synergy index peaks at moderate force')]
    for ax, (key, col, ylab, title) in zip(axs, panels):
        ax.plot(d['meanF'], d[key], '-o', color=col, lw=2, ms=5)
        ax.set_xlabel('mean total force [a.u.]')
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=.35)
    fig.suptitle('Sensitivity to force level (1 Hz task)', fontsize=13)
    fig.tight_layout()
    save(fig, 'force_level.png')


FIGS = {1: fig1_pca, 2: fig2_bars, 3: fig3_scatter, 4: fig4_cellwise,
        5: fig5_sweep, 6: fig6_ucm, 7: fig7_rates, 8: fig8_force_level}


if __name__ == '__main__':
    wanted = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else sorted(FIGS)
    for n in wanted:
        print(f'Figure {n}')
        FIGS[n]()
