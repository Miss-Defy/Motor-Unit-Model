"""Spinal circuitry as the basis for motor unit-based synergies.

The model and every analysis it feeds. Figures live in figures.py.

A pool of five Hindmarsh-Rose alpha-motoneurons in two MU-modes
(MU1-3 and MU4-5), five Renshaw cells, two Ib interneurons and Golgi tendon
organ feedback, driven by a sinusoidal descending drive. Renshaw connectivity
is complete: every Renshaw cell inhibits every alpha-MN including its own and
no gain is zero, so the modes are defined by the ratio of the gains
(wR_cross / wR_within = 10) rather than by which connections exist.

Task matched to Madarshahian et al. 2021 (J Physiol) and Ricotta et al. 2023
(JAP): 1 Hz sinusoidal force target, cycle divided into 20 windows of 5%,
f_MU smoothed with a 1 s Hann window, MU firing rates of 5 to 20 Hz.
Delays: Renshaw 5 ms, Golgi/Ib 50 ms, on the interneuron -> alpha-MN arm.
HR dynamics are time-scaled by tau_hr so firing rates are physiological.

Contents
--------
    run()               integrate the model, return spikes, force, drive
    analyse()           bin by phase, the two correlation analyses
    hann_rate()         spike train -> firing frequency
    rates_hz()          mean firing rates of each cell type
    cfg()               task configuration for a given cycle duration
    ucm_indices()       V_UCM / V_ORT / dV, three ways of identifying J
    pca_modes()         PCA with varimax rotation
    windows()           f_MU in 5% phase windows, whole or half cycle
    varimax()
    sweep_gains()       2-D sweep over the two inhibitory gains
    sweep_force()       sweep over mean descending drive
    sweep_rates()       indices as a function of task rate
    sweep_force_gains() indices as a function of the spread of b_i
    ablations()         contributions of the two inhibitory pathways

Running `python model.py` writes everything figures.py reads into data/:

    data/cases.npz      per-rate arrays (binned rates, UCM, PCA loadings)
    data/cases.json     the scalar indices quoted in the text
    data/sweep_gains.npz   the 2-D inhibition-gain sweep   (--sweeps)
    data/rates.json        indices against task rate       (--sweeps)
    data/force_sweep.json  indices against force level     (--sweeps)
    data/tables.json       ablations and force-gain spread (--sweeps)

The default run does the two task rates only, which takes about a minute.
`--sweeps` adds the three sweeps and the tables, which take tens of minutes.
"""
import json
import os as _os

import numpy as np
from numba import njit
from scipy.signal import lfilter, find_peaks, get_window
from scipy.stats import pearsonr

A_HR, B_HR, C_HR, D_HR = 1.0, 3.0, 1.0, 5.0
R_HR, S_HR, XR_HR = 0.005, 4.0, -1.6


@njit(cache=True, fastmath=True)
def _sim(T, dt, dt_out, N_MU, mode_of, tau_hr,
         drive_base, drive_mod, drive_phase, drive_amp,
         mode_share, unit_noise, share_base, force_gain,
         gRC, W_Ib, sense_mask, tau_act, k_F, Ib_bias, I_Ib_base, w_aff,
         Ia_drive, w_Ia, I_R_base, w_MN_R,
         alpha_syn, tau_syn, v_th, k_s, lag_rc, lag_ib, ib_sub, y0):

    n_neu = 2 * N_MU + 2
    n_st = 4 * n_neu + N_MU
    n_out = int(T / dt_out)
    spo = int(dt_out / dt)
    aoff = 4 * n_neu
    inv_tau = 1.0 / tau_hr

    st = y0.copy()
    spk = np.zeros((n_neu, n_out), np.int32)
    Frec = np.zeros((N_MU, n_out))
    IbAff = np.zeros((2, n_out))
    cnsrec = np.zeros(n_out)
    prev_v = np.zeros(n_neu)
    for k in range(n_neu):
        prev_v[k] = st[4 * k]

    # ring buffer of synaptic gating variables, for conduction delays
    maxlag = lag_rc if lag_rc > lag_ib else lag_ib
    maxlag = maxlag + 2
    sbuf = np.zeros((n_neu, maxlag))
    bpos = 0

    d = np.zeros(n_st)
    k1 = np.zeros(n_st); k2 = np.zeros(n_st); k3 = np.zeros(n_st); k4 = np.zeros(n_st)
    tmp = np.zeros(n_st)
    Iarr = np.zeros(n_neu)
    Fi = np.zeros(N_MU)
    Fs = np.zeros(2)
    sRC = np.zeros(N_MU)
    sIb = np.zeros(2)

    for i in range(n_out):
        t0 = i * dt_out
        for j in range(spo):
            t = t0 + j * dt

            # delayed interneuron outputs, held across this dt step
            irc = bpos - lag_rc
            while irc < 0:
                irc += maxlag
            iib = bpos - lag_ib
            while iib < 0:
                iib += maxlag
            for m in range(N_MU):
                sRC[m] = sbuf[N_MU + m, irc]
            for kk in range(2):
                sIb[kk] = sbuf[2 * N_MU + kk, iib]

            for stage in range(4):
                if stage == 0:
                    for q in range(n_st): tmp[q] = st[q]
                    tt = t
                elif stage == 1:
                    for q in range(n_st): tmp[q] = st[q] + 0.5 * dt * k1[q]
                    tt = t + 0.5 * dt
                elif stage == 2:
                    for q in range(n_st): tmp[q] = st[q] + 0.5 * dt * k2[q]
                    tt = t + 0.5 * dt
                else:
                    for q in range(n_st): tmp[q] = st[q] + dt * k3[q]
                    tt = t + dt

                # Phase and amplitude are precomputed per output step, so the
                # cycle period and the cycle amplitude can both vary.
                cns = drive_amp[i] * np.sin(drive_phase[i])
                cns_norm = 0.5 * (cns + 1.0)
                D = drive_base + drive_mod * cns_norm

                for m in range(N_MU):
                    Fi[m] = force_gain[m] * tmp[aoff + m]

                for kk in range(2):
                    acc = 0.0
                    for m in range(N_MU):
                        acc += sense_mask[kk, m] * Fi[m]
                    v = k_F * acc + Ib_bias
                    Fs[kk] = v if v > 0.0 else 0.0
                    IbAff[kk, i] = Fs[kk]

                # ---- alpha motor neurons (delayed inhibition) ----
                for m in range(N_MU):
                    share = share_base[m] * mode_share[mode_of[m], i]
                    inh = 0.0
                    for q in range(N_MU):
                        inh += gRC[m, q] * sRC[q]
                    for kk in range(2):
                        inh += W_Ib[m, kk] * sIb[kk]
                    Iarr[m] = D * share + w_Ia * Ia_drive + unit_noise[m, i] - inh
                for m in range(N_MU):
                    Iarr[N_MU + m] = I_R_base + w_MN_R * tmp[4 * m + 3]
                for kk in range(2):
                    if ib_sub == 1:
                        Iarr[2 * N_MU + kk] = I_Ib_base      # held below threshold
                    else:
                        Iarr[2 * N_MU + kk] = I_Ib_base + w_aff * Fs[kk]

                for k in range(n_neu):
                    x = tmp[4 * k]; y = tmp[4 * k + 1]
                    z = tmp[4 * k + 2]; sv = tmp[4 * k + 3]
                    Ik = Iarr[k]
                    d[4 * k]     = inv_tau * (y - A_HR*x*x*x + B_HR*x*x - z + Ik)
                    d[4 * k + 1] = inv_tau * (C_HR - D_HR*x*x - y)
                    d[4 * k + 2] = inv_tau * (R_HR * (S_HR*(x - XR_HR) - z))
                    if ib_sub == 1 and k >= 2 * N_MU:
                        # Subthreshold Ib interneuron: graded transmission, the
                        # membrane signal tracking Ib afferent rate without spiking.
                        a_ = Fs[k - 2 * N_MU]
                        H = a_ / (1.0 + a_)
                    else:
                        H = 1.0 / (1.0 + np.exp(-(x - v_th) / k_s))
                    d[4 * k + 3] = alpha_syn * H * (1.0 - sv) - sv / tau_syn

                for m in range(N_MU):
                    d[aoff + m] = (tmp[4 * m + 3] - tmp[aoff + m]) / tau_act

                if stage == 0:
                    for q in range(n_st): k1[q] = d[q]
                elif stage == 1:
                    for q in range(n_st): k2[q] = d[q]
                elif stage == 2:
                    for q in range(n_st): k3[q] = d[q]
                else:
                    for q in range(n_st): k4[q] = d[q]

            for q in range(n_st):
                st[q] += dt / 6.0 * (k1[q] + 2.0*k2[q] + 2.0*k3[q] + k4[q])

            bpos += 1
            if bpos >= maxlag:
                bpos = 0
            for k in range(n_neu):
                sbuf[k, bpos] = st[4 * k + 3]

            for k in range(n_neu):
                v = st[4 * k]
                if prev_v[k] <= 0.0 and v > 0.0:
                    spk[k, i] += 1
                prev_v[k] = v

        for m in range(N_MU):
            Frec[m, i] = force_gain[m] * st[aoff + m]
        cnsrec[i] = drive_amp[i] * np.sin(drive_phase[i])

    return spk, Frec, IbAff, cnsrec


def ou(n, dt_out, tau, sigma, rng):
    a = np.exp(-dt_out / tau)
    return lfilter([sigma * np.sqrt(1 - a * a)], [1.0, -a], rng.standard_normal(n))


DEF = dict(
    N_MU=5, mode_of=[0, 0, 0, 1, 1], force_gain=[0.9, 1.0, 1.1, 1.0, 1.2],
    mode_frac=[0.45, 0.55],
    share_scheme='per_unit',
    unit_share=[0.19, 0.20, 0.21, 0.19, 0.21],
    # --- real time ---
    dt=0.05, dt_out=1.0,            # ms
    cycle_ms=1000.0,                # 1 Hz task (mean period)
    period_cv=0.05,                 # SD of cycle duration, as a fraction of the mean
    amp_cv=0.05,                    # SD of cycle amplitude, as a fraction of the mean
    gain_cv=0.10,                   # SD of each synaptic weight about its nominal value
    n_cycles=45, t_skip_ms=2000.0,
    tau_hr=7.4,                     # ms per intrinsic HR unit
    # --- drive ---
    drive_base=5.0, drive_mod=3.0,
    Ia_drive=1.0, w_Ia=0.3,
    # --- Renshaw ---
    normalise_rc=True,
    I_R_base=1.0, w_MN_R=3.0, wR_self=0.3, wR_within=0.3, wR_cross=3.0,
    delay_rc_ms=5.0,
    # --- force / Golgi / Ib ---
    tau_act_ms=100.0, k_F=1.0, Ib_bias=0.0, I_Ib_base=1.0, w_aff=5.5,
    gto_mode='total', proj_mode=None, normalise_gto=True,
    ib_subthreshold=False,
    w_Ib=0.5, w_Ib_cross=0.0, delay_ib_ms=50.0,
    # --- synapses / noise (ms) ---
    alpha_syn=1.0, tau_syn_ms=100.0, v_th=0.0, k_s=0.3,
    tau_noise_ms=150.0, unit_noise_sd=0.6,
    tau_share_ms=150.0, share_noise_sd=0.10,
    constrained_sharing=False,
    # --- analysis ---
    n_bins=20, hann_ms=1000.0, seed=1)


def run(**kw):
    p = dict(DEF); p.update(kw)
    rng = np.random.default_rng(p['seed'])
    N = p['N_MU']
    mode_of = np.array(p['mode_of'], np.int64)
    # ---- per-cycle duration and amplitude, drawn from normal distributions ----
    ncyc_tot = int(p['n_cycles'] + p['t_skip_ms'] / p['cycle_ms']) + 3
    periods = p['cycle_ms'] * (1.0 + p['period_cv'] * rng.standard_normal(ncyc_tot))
    periods = np.clip(periods, 0.5 * p['cycle_ms'], 1.5 * p['cycle_ms'])
    amps = 1.0 + p['amp_cv'] * rng.standard_normal(ncyc_tot)
    amps = np.clip(amps, 0.2, 1.8)

    T = p['n_cycles'] * p['cycle_ms'] + p['t_skip_ms']
    n_out = int(T / p['dt_out'])

    # phase accumulates at a rate set by the current cycle's period, so cycle
    # boundaries fall where the phase passes a multiple of 2*pi
    t_grid = np.arange(n_out) * p['dt_out']
    edges = np.concatenate([[0.0], np.cumsum(periods)])
    cyc_idx = np.clip(np.searchsorted(edges, t_grid, 'right') - 1, 0, ncyc_tot - 1)
    frac = (t_grid - edges[cyc_idx]) / periods[cyc_idx]
    drive_phase = 2.0 * np.pi * frac
    drive_amp = amps[cyc_idx]

    frac = np.array(p['mode_frac'], float)
    if p['share_scheme'] == 'per_unit':
        # Slide 7 defines a_i PER MOTOR UNIT with sum(a_i) = 1. Near-equal
        # shares, with a small spread standing in for size-principle
        # differences between units.
        share_base = np.array(p['unit_share'], float)
    else:
        # 'mode_fraction': each mode's share split evenly among ITS units.
        # With 3 vs 2 units this gives mode-2 units ~1.8x the drive, which is
        # an artefact of unequal group size rather than a modelling claim.
        counts = np.array([(mode_of == k).sum() for k in range(len(frac))], float)
        share_base = np.array([frac[mode_of[m]] / counts[mode_of[m]] for m in range(N)])
    share_base = share_base * N / share_base.sum()

    xi = ou(n_out, p['dt_out'], p['tau_share_ms'], p['share_noise_sd'], rng)
    ms_ = np.empty((2, n_out))
    ms_[0] = 1.0 + xi
    if p['constrained_sharing']:
        ms_[1] = 1.0 - xi * frac[0] / frac[1]
    else:
        ms_[1] = 1.0 + ou(n_out, p['dt_out'], p['tau_share_ms'], p['share_noise_sd'], rng)
    ms_ = np.clip(ms_, 0.05, None)

    unit_noise = np.vstack([ou(n_out, p['dt_out'], p['tau_noise_ms'],
                               p['unit_noise_sd'], rng) for _ in range(N)])

    def jitter(x):
        """Draw a synaptic weight around its nominal value (gain_cv = 0 -> fixed)."""
        if p['gain_cv'] <= 0:
            return x
        return max(0.0, x * (1.0 + p['gain_cv'] * rng.standard_normal()))

    gRC = np.empty((N, N))
    nmode = np.array([(mode_of == k).sum() for k in range(2)], float)
    for a_ in range(N):
        for b_ in range(N):
            if a_ == b_:
                gRC[a_, b_] = jitter(p['wR_self'])
            elif mode_of[a_] == mode_of[b_]:
                g = p['wR_within']
                # Weights name the TOTAL inhibition arriving from a mode,
                # divided among its cells. Without this a unit in the smaller
                # mode receives more cross-mode inhibition simply because the
                # other mode has more cells, which distorts its drive-following.
                if p['normalise_rc']:
                    g /= max(nmode[mode_of[b_]] - 1.0, 1.0)
                gRC[a_, b_] = jitter(g)
            else:
                g = p['wR_cross']
                if p['normalise_rc']:
                    g /= nmode[mode_of[b_]]
                gRC[a_, b_] = jitter(g)

    sense = np.zeros((2, N)); W_Ib = np.zeros((N, 2))
    if p['gto_mode'] == 'total':
        sense[:, :] = 1.0
    else:
        for k in range(2):
            sense[k, mode_of == k] = 1.0
    pm = p.get('proj_mode') or ('all' if p['gto_mode'] == 'total' else 'per_mode')
    if pm == 'all':
        for m in range(N):
            for k in range(2):
                W_Ib[m, k] = jitter(p['w_Ib'])
    else:
        for m in range(N):
            for k in range(2):
                W_Ib[m, k] = jitter(p['w_Ib'] if mode_of[m] == k else p['w_Ib_cross'])
    if p['normalise_gto']:
        sense /= sense.sum(1, keepdims=True)

    n_neu = 2 * N + 2
    y0 = np.zeros(4 * n_neu + N)
    for k in range(n_neu):
        y0[4 * k] = -1.6 + 0.01 * rng.standard_normal()
        y0[4 * k + 1] = -12.0

    lag_rc = max(int(round(p['delay_rc_ms'] / p['dt'])), 1)
    lag_ib = max(int(round(p['delay_ib_ms'] / p['dt'])), 1)

    spk, F, IbAff, cns = _sim(
        T, p['dt'], p['dt_out'], N, mode_of, p['tau_hr'],
        p['drive_base'], p['drive_mod'], drive_phase, drive_amp,
        ms_, unit_noise, share_base, np.array(p['force_gain'], float),
        gRC, W_Ib, sense, p['tau_act_ms'], p['k_F'], p['Ib_bias'],
        p['I_Ib_base'], p['w_aff'], p['Ia_drive'], p['w_Ia'],
        p['I_R_base'], p['w_MN_R'], p['alpha_syn'], p['tau_syn_ms'],
        p['v_th'], p['k_s'], lag_rc, lag_ib,
        1 if p['ib_subthreshold'] else 0, y0)

    p['_T'] = T
    return spk, F, IbAff, cns, p


def hann_rate(spk_row, dt_out, hann_ms):
    """f_MU via a Hann window, as in Madarshahian 2021 / Ricotta 2023. -> Hz"""
    L = int(round(hann_ms / dt_out))
    if L % 2 == 0:
        L += 1
    w = get_window('hann', L)
    w = w / w.sum()
    return np.convolve(spk_row.astype(float), w, mode='same') * (1000.0 / dt_out)


def analyse(spk, F, cns, p):
    N = p['N_MU']; dt_out = p['dt_out']; nb = p['n_bins']
    mode_of = np.array(p['mode_of'])

    fMU = np.vstack([hann_rate(spk[m], dt_out, p['hann_ms']) for m in range(N)])
    M = np.vstack([fMU[mode_of == k].sum(0) for k in (0, 1)])     # mode freq sums

    troughs, _ = find_peaks(-cns, distance=int(0.6 * p['cycle_ms'] / dt_out))
    troughs = np.array([t for t in troughs if t * dt_out > p['t_skip_ms']])
    # Reject cycles whose duration departs from nominal by more than 20%,
    # following Madarshahian et al. (2021).
    if len(troughs) > 1:
        dur = np.diff(troughs) * dt_out
        keep = np.abs(dur - p['cycle_ms']) <= 0.2 * p['cycle_ms']
        pairs = [(troughs[i], troughs[i+1]) for i in range(len(dur)) if keep[i]]
    else:
        pairs = []

    binned, unit_binned = [], []
    for c in range(len(troughs) - 1):
        e = np.linspace(troughs[c], troughs[c + 1], nb + 1).astype(int)
        binned.append([[M[k, e[j]:e[j+1]].mean() for k in (0, 1)] for j in range(nb)])
        unit_binned.append([[fMU[m, e[j]:e[j+1]].mean() for m in range(N)]
                            for j in range(nb)])
    binned = np.array(binned); unit_binned = np.array(unit_binned)

    bm = binned.mean(0)
    sl_a = np.polyfit(bm[:, 0], bm[:, 1], 1)[0]
    r_a = np.corrcoef(bm[:, 0], bm[:, 1])[0, 1]
    cns_bin = np.sin(2*np.pi*(np.arange(nb)+0.5)/nb - np.pi/2)
    g1 = (pearsonr(bm[:, 0], cns_bin)[0], pearsonr(bm[:, 1], cns_bin)[0])

    slopes = np.zeros(nb); rr = np.zeros(nb); ucm = np.zeros(nb); ort = np.zeros(nb)
    for j in range(nb):
        x, y = binned[:, j, 0], binned[:, j, 1]
        slopes[j] = np.polyfit(x, y, 1)[0]
        rr[j] = np.corrcoef(x, y)[0, 1]
        ort[j] = np.var((x + y) / np.sqrt(2))
        ucm[j] = np.var((x - y) / np.sqrt(2))

    return dict(binned=binned, unit_binned=unit_binned, n_cycles=binned.shape[0],
                bin_mean=bm, M=M, fMU=fMU, troughs=troughs, g1=g1,
                slope_across_bins=sl_a, r_across_bins=r_a,
                slopes=slopes, r=rr, ucm=ucm, ort=ort)


def rates_hz(spk, p):
    N = p['N_MU']; sk = int(p['t_skip_ms'] / p['dt_out'])
    dur_s = (p['_T'] - p['t_skip_ms']) / 1000.0
    return (spk[:N, sk:].sum() / N / dur_s,
            spk[N:2*N, sk:].sum() / N / dur_s,
            spk[2*N:, sk:].sum() / 2 / dur_s)


def summarise(tag, res, spk, p):
    mn, rc, ib = rates_hz(spk, p)
    print(f"[{tag}] cycles={res['n_cycles']}  MN={mn:.1f} Hz  RC={rc:.1f} Hz  IbIN={ib:.1f} Hz")
    print(f"   1a across bins : r={res['r_across_bins']:+.2f}   "
          f"r(f,CNS)={res['g1'][0]:+.2f}/{res['g1'][1]:+.2f}")
    print(f"   1b within bin  : b={res['slopes'].mean():+.2f}  r={res['r'].mean():+.2f}  "
          f"neg={int((res['slopes'] < 0).sum())}/{len(res['slopes'])}")
    print(f"   UCM/ORT        : {np.mean(res['ucm']/res['ort']):.2f}")


# --------------------------------------------------------------------------
# task configuration
# --------------------------------------------------------------------------
def cfg(cyc, **kw):
    """Parameters for a task of cycle duration `cyc` ms.

    Neural constants are fixed across rates; task-related constants (noise
    correlation time, sharing fluctuation time, smoothing window) scale with
    cycle duration, as they would in an experiment.
    """
    s = cyc / 1000.0
    d = dict(cycle_ms=cyc, tau_noise_ms=150 * s, tau_share_ms=150 * s,
             hann_ms=1000 * s, t_skip_ms=2 * cyc, n_cycles=45,
             drive_base=5.0, drive_mod=3.0, normalise_rc=True,
             wR_cross=3.0, w_Ib=0.5, seed=1)
    d.update(kw)
    return d


# --------------------------------------------------------------------------
# UCM decomposition
# --------------------------------------------------------------------------
def ucm_indices(spk, F, cns, p, jac='analytic'):
    """Mode magnitudes in f_MU space, performance variable = total force.

    The Jacobian is identified per phase bin by regressing F_TOT on the two
    mode magnitudes across cycles, as done experimentally -- the modes live in
    firing-frequency space while the performance variable is force, so the
    mapping is not the identity.
    """
    N = p['N_MU']; dt_out = p['dt_out']; nb = p['n_bins']
    mode_of = np.array(p['mode_of'])
    fMU = np.vstack([hann_rate(spk[i], dt_out, p['hann_ms']) for i in range(N)])
    M = np.vstack([fMU[mode_of == k].sum(0) for k in (0, 1)])
    Ftot = F.sum(0)

    tr, _ = find_peaks(-cns, distance=int(0.8 * p['cycle_ms'] / dt_out))
    tr = [t for t in tr if t * dt_out > p['t_skip_ms']]

    Mb, Fb = [], []
    for c in range(len(tr) - 1):
        e = np.linspace(tr[c], tr[c+1], nb+1).astype(int)
        Mb.append([[M[k, e[j]:e[j+1]].mean() for k in (0, 1)] for j in range(nb)])
        Fb.append([Ftot[e[j]:e[j+1]].mean() for j in range(nb)])
    Mb = np.array(Mb); Fb = np.array(Fb)          # (cycles, bins, 2), (cycles, bins)

    # Over-cycle multiple linear regression, as used experimentally: all phase
    # windows and cycles pooled into one fit, rather than a separate fit per window.
    Xa = Mb.reshape(-1, 2); ya = Fb.reshape(-1)
    Xa = Xa - Xa.mean(0); ya = ya - ya.mean()
    J_over, *_ = np.linalg.lstsq(Xa, ya, rcond=None)

    Vucm = np.zeros(nb); Vort = np.zeros(nb)
    dV = np.zeros(nb); dVz = np.zeros(nb); Jn = np.zeros((nb, 2))
    for j in range(nb):
        X = Mb[:, j, :] - Mb[:, j, :].mean(0)
        y = Fb[:, j] - Fb[:, j].mean()
        if jac == 'within_bin':
            J, *_ = np.linalg.lstsq(X, y, rcond=None)
        elif jac == 'overcycle':
            J = J_over                            # fitted once over the whole cycle
        else:
            # Analytic. F = sum_i b_i a_i, and a_i is a shared monotone
            # function of f_i, so dF/dM_k is proportional to the MEAN force
            # gain of mode k. Regression identification is ill-conditioned
            # here because the two modes are strongly correlated.
            bm = np.array([np.mean(np.array(p['force_gain'])[mode_of == k])
                           for k in (0, 1)])
            J = bm
        Jn[j] = J
        u = J / np.linalg.norm(J)                       # ORT direction
        v = np.array([-u[1], u[0]])                     # UCM direction (null space)
        # one DOF each, so the per-DOF variances are the projections themselves
        Vort[j] = np.var(X @ u)
        Vucm[j] = np.var(X @ v)
        tot = (Vucm[j] + Vort[j]) / 2.0
        dV[j] = (Vucm[j] - Vort[j]) / tot               # bounded by +/-2
        q = np.clip(dV[j] / 2.0, -0.999, 0.999)
        dVz[j] = 0.5 * np.log((1 + q) / (1 - q))        # Fisher z
    return dict(Vucm=Vucm, Vort=Vort, dV=dV, dVz=dVz, J=Jn, Mb=Mb, Fb=Fb)


# --------------------------------------------------------------------------
# PCA identification of MU-modes
# --------------------------------------------------------------------------
def varimax(Phi, gamma=1.0, q=200, tol=1e-7):
    p, k = Phi.shape
    R = np.eye(k); dd = 0
    if k < 2:
        return Phi, R
    for _ in range(q):
        d_old = dd
        L = Phi @ R
        u, s, vh = np.linalg.svd(
            Phi.T @ (L**3 - (gamma/p) * L @ np.diag(np.diag(L.T @ L))))
        R = u @ vh; dd = np.sum(s)
        if d_old != 0 and dd/d_old < 1 + tol:
            break
    return Phi @ R, R


def pca_modes(X, nmax=None, force_k=None):
    """X: (observations, units). Returns eigenvalues, % variance, varimax loadings."""
    Z = (X - X.mean(0)) / X.std(0)
    C = np.corrcoef(Z, rowvar=False)
    ev, V = np.linalg.eigh(C)
    idx = np.argsort(ev)[::-1]
    ev, V = ev[idx], V[:, idx]
    pct = 100 * ev / ev.sum()
    k_crit = int((ev >= 1.0).sum())          # eigenvalue >= 1 criterion
    k = force_k if force_k else k_crit
    if nmax:
        k = min(k, nmax)
    k = max(k, 1)
    load = V[:, :k] * np.sqrt(ev[:k])          # unrotated loadings
    rot, _ = varimax(load)
    # sign convention: make each factor's largest loading positive
    for j in range(rot.shape[1]):
        if rot[np.argmax(np.abs(rot[:, j])), j] < 0:
            rot[:, j] *= -1
    # order factors by which units they load, so panels are comparable
    order = np.argsort([np.argmax(np.abs(rot[:, j])) for j in range(rot.shape[1])])
    return ev, pct, rot[:, order], k_crit


def windows(spk, cns, p, nwin=20, half=None):
    """Mean f_MU in equal windows. half=None whole cycle; 'up'/'down' half-cycles."""
    N = p['N_MU']; dt_out = p['dt_out']
    fMU = np.vstack([hann_rate(spk[i], dt_out, p['hann_ms']) for i in range(N)])
    dist = int(0.8 * p['cycle_ms'] / dt_out)
    tr, _ = find_peaks(-cns, distance=dist)
    pk, _ = find_peaks(cns, distance=dist)
    tr = np.array([t for t in tr if t * dt_out > p['t_skip_ms']])
    pk = np.array([t for t in pk if t * dt_out > p['t_skip_ms']])
    rows = []
    for c in range(len(tr) - 1):
        a, b = tr[c], tr[c + 1]
        if half is None:
            seg = [(a, b)]
        else:
            mid = pk[(pk > a) & (pk < b)]
            if len(mid) == 0:
                continue
            mid = mid[0]
            seg = [(a, mid)] if half == 'up' else [(mid, b)]
        for (s0, s1) in seg:
            e = np.linspace(s0, s1, nwin + 1).astype(int)
            for j in range(nwin):
                if e[j + 1] > e[j]:
                    rows.append(fMU[:, e[j]:e[j + 1]].mean(1))
    return np.array(rows)


# --------------------------------------------------------------------------
# sweeps and ablations
# --------------------------------------------------------------------------
def _indices(cyc, seeds=(1, 2, 3), **kw):
    """Mean +/- SD across seeds of the indices quoted throughout."""
    a = []
    for sd in seeds:
        spk, F, I, cns, p = run(**cfg(cyc, seed=sd, **kw))
        r = analyse(spk, F, cns, p)
        u = ucm_indices(spk, F, cns, p)
        mn, _, _ = rates_hz(spk, p)
        a.append((mn, r['r_across_bins'], r['r'].mean(),
                  (r['slopes'] < 0).sum(), np.mean(r['ucm'] / r['ort']),
                  u['dV'].mean()))
    a = np.array(a)
    return a.mean(0), a.std(0)


def _one_gain_point(args):
    """One cell of the 2-D gain sweep. Top level so it can be pickled."""
    gm, i, j, wr, wib, ncyc = args
    spk, F, IbAff, cns, p = run(n_cycles=ncyc, gto_mode=gm,
                                wR_cross=float(wr), w_Ib=float(wib), seed=1)
    r = analyse(spk, F, cns, p)
    mn, rc, ib = rates_hz(spk, p)
    return (gm, i, j, float(r['r'].mean()),
            float(np.mean(r['ucm'] / r['ort'])), float(mn),
            float(r['r_across_bins']), int((r['slopes'] < 0).sum()),
            float(min(r['g1'])))


def sweep_gains(n=25, ncyc=40, nproc=2):
    """Renshaw between-mode gain x Ib gain, for both Golgi scenarios.

    1250 simulations at the default resolution; this is the slow one.
    Returns a dict of (n, n) grids suitable for np.savez.
    """
    from multiprocessing import Pool
    WR = np.round(np.linspace(0.0, 6.0, n), 3)
    WIB = np.round(np.linspace(0.0, 4.0, n), 3)
    gtos = ['total', 'per_mode']
    jobs = [(gm, i, j, WR[i], WIB[j], ncyc)
            for gm in gtos for i in range(n) for j in range(n)]
    print(f'{len(jobs)} runs ...', flush=True)
    keys = ['R', 'U', 'MN', 'RA', 'NG', 'G1']
    A = {k: {g: np.zeros((n, n)) for g in gtos} for k in keys}
    with Pool(nproc) as pool:
        for c, res in enumerate(pool.imap_unordered(_one_gain_point, jobs,
                                                    chunksize=4), 1):
            gm, i, j = res[0], res[1], res[2]
            for k, v in zip(keys, res[3:]):
                A[k][gm][i, j] = v
            if c % 50 == 0:
                print(f'  {c}/{len(jobs)}', flush=True)
    out = {'WR': WR, 'WIB': WIB}
    out.update({f'{k}_{g}': A[k][g] for k in keys for g in gtos})
    return out


def sweep_force(bases=None, mod_frac=0.6, n_cycles=45, seed=1):
    """Mean descending drive against force variability and the synergy index."""
    bases = bases if bases is not None else list(np.arange(2.0, 8.75, 0.5))
    rows = []
    for base in bases:
        spk, F, I, cns, p = run(**cfg(1000.0, n_cycles=n_cycles, seed=seed,
                                      drive_base=base,
                                      drive_mod=mod_frac * base))
        r = analyse(spk, F, cns, p)
        u = ucm_indices(spk, F, cns, p)
        mn, _, _ = rates_hz(spk, p)
        Ftot = F.sum(0)
        meanF, sdF = float(Ftot.mean()), float(Ftot.std())
        rows.append([float(mn), meanF, sdF, 100.0 * sdF / meanF,
                     float(r['r'].mean()), float(u['dV'].mean())])
        print(f'  base={base:4.1f}  {mn:5.2f} Hz  F={meanF:.3f}+-{sdF:.3f}  '
              f'r={rows[-1][4]:+.2f}  dV={rows[-1][5]:+.2f}', flush=True)
    return dict(base=list(map(float, bases)),
                cols=['MNHz', 'meanF', 'sdF', 'cvpct', 'r1b', 'dV'], data=rows)


def sweep_rates(cycles=(1000., 2000., 3000., 5000., 10000.), seeds=(1, 2, 3)):
    """Indices of co-variation as a function of task rate."""
    rows = []
    for cyc in cycles:
        mu, sd = _indices(cyc, seeds)
        rows.append([cyc / 1000., round(float(mu[2]), 3), round(float(mu[4]), 3),
                     round(float(mu[5]), 3), round(float(sd[2]), 3),
                     round(float(sd[4]), 3), round(float(sd[5]), 3)])
        print(f'  {cyc/1000:5.1f} s  r={mu[2]:+.2f}  UCM={mu[4]:.2f}  '
              f'dV={mu[5]:+.2f}', flush=True)
    return rows


FORCE_GAIN_SETS = {
    'narrow (0.9-1.2, as used)': [0.9, 1.0, 1.1, 1.0, 1.2],
    'wide (0.5-2.7, ~5x)':       [0.5, 0.8, 1.2, 1.8, 2.7],
    'very wide (0.3-4.0, ~13x)': [0.3, 0.6, 1.1, 2.1, 4.0],
}


def sweep_force_gains(sets=None, seeds=(1, 2, 3)):
    """Does the spread of per-unit force gains matter? (It does not.)"""
    sets = sets or FORCE_GAIN_SETS
    out = {}
    for lbl, g in sets.items():
        mu, sd = _indices(1000.0, seeds, force_gain=g)
        out[lbl] = [round(float(x), 3) for x in
                    (mu[2], mu[4], mu[5], sd[2], sd[4], sd[5])]
        print(f'  {lbl:28s} r={mu[2]:+.2f}  UCM={mu[4]:.2f}  dV={mu[5]:+.2f}',
              flush=True)
    return out


ABLATION_CONDS = [(3.0, 0.5, 'Renshaw + Ib'), (3.0, 0.0, 'Renshaw only'),
                  (0.0, 0.5, 'Ib only (w = 0.5)'), (0.0, 2.0, 'Ib only (w = 2.0)'),
                  (0.0, 0.0, 'Neither')]


def ablations(cycles=((1000., '1Hz'), (10000., 'slow')), seeds=(1, 2, 3, 4)):
    """Switch each inhibitory pathway off in turn, at each task rate."""
    out = {}
    for cyc, tag in cycles:
        for rx, wi, lbl in ABLATION_CONDS:
            mu, sd = _indices(cyc, seeds, wR_cross=rx, w_Ib=wi)
            out[f'{tag}|{lbl}'] = [round(float(x), 2) for x in
                                   (mu[0], mu[1], mu[2], sd[2], mu[3], mu[4], sd[4])]
            print(f'  {tag:5s} {lbl:18s} MU={mu[0]:5.1f}  r={mu[2]:+.2f}+-{sd[2]:.2f}  '
                  f'neg={mu[3]:4.1f}/20  UCM={mu[4]:.2f}+-{sd[4]:.2f}', flush=True)
    return out


# --------------------------------------------------------------------------
# producing the stored results
# --------------------------------------------------------------------------
HERE = _os.path.dirname(_os.path.abspath(__file__))
DATA = _os.path.join(HERE, 'data')

FAST, SLOW = 1000.0, 10000.0
TAGS = {FAST: 'fast', SLOW: 'slow'}


def run_case(cyc):
    """One task rate: everything the figures need from it."""
    spk, F, I, cns, p = run(**cfg(cyc))
    res = analyse(spk, F, cns, p)
    u = ucm_indices(spk, F, cns, p, jac='analytic')
    uo = ucm_indices(spk, F, cns, p, jac='overcycle')
    mn, rc, ib = rates_hz(spk, p)

    arrays = {
        'binned': res['binned'],              # (cycles, windows, 2)
        'unit_binned': res['unit_binned'],    # (cycles, windows, 5)
        'bin_mean': res['bin_mean'],          # (windows, 2)
        'r': res['r'], 'slopes': res['slopes'],
        'ucm': res['ucm'], 'ort': res['ort'],
        'Vucm': u['Vucm'], 'Vort': u['Vort'], 'dV': u['dV'],
        'g1': np.array(res['g1']),
    }
    scalars = {
        'cycle_s': cyc / 1000.0, 'n_bins': int(p['n_bins']),
        'n_cycles': int(res['n_cycles']), 'mu_rate_hz': float(mn),
        'rc_rate_hz': float(rc), 'ib_rate_hz': float(ib),
        'r_across_bins': float(res['r_across_bins']),
        'r_within_bin': float(res['r'].mean()),
        'n_negative': int((res['slopes'] < 0).sum()),
        'ucm_ratio': float(np.mean(res['ucm'] / res['ort'])),
        'dV': float(u['dV'].mean()), 'dV_overcycle': float(uo['dV'].mean()),
    }

    # PCA identification of the modes, whole cycle and each half-cycle
    X = windows(spk, cns, p, 20)
    ev, pct, L, _ = pca_modes(X, nmax=2)
    arrays['pca_loadings'] = L
    arrays['pca_eigenvalues'] = ev
    scalars['pca_pct'] = float(pct[:2].sum())
    for hv in ('up', 'down'):
        Xh = windows(spk, cns, p, 10, half=hv)
        evh, pcth, Lh, _ = pca_modes(Xh, nmax=2, force_k=2)
        arrays[f'pca_loadings_{hv}'] = Lh
        scalars[f'pca_pct_{hv}'] = float(pcth[:2].sum())
    scalars['pca_half_corr'] = [
        float(abs(np.corrcoef(arrays['pca_loadings_up'][:, j],
                              arrays['pca_loadings_down'][:, j])[0, 1]))
        for j in range(2)]

    print(f'  {cyc/1000:.0f} s cycle: MU={mn:.1f} Hz  '
          f'r across windows={scalars["r_across_bins"]:+.2f}  '
          f'r within window={scalars["r_within_bin"]:+.2f}  '
          f'{scalars["n_negative"]}/{p["n_bins"]} negative  '
          f'dV={scalars["dV"]:+.2f} (over-cycle J {scalars["dV_overcycle"]:+.2f})',
          flush=True)
    return arrays, scalars


def main(argv=None):
    """Run the model and write everything the figures read into data/."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sweeps', action='store_true',
                    help='also run the slow sweeps (gains, task rate, force level)')
    ap.add_argument('--only', choices=['cases', 'gains', 'rates', 'force',
                                       'tables'], nargs='*',
                    help='run only these parts')
    a = ap.parse_args(argv)
    parts = a.only or (['cases', 'gains', 'rates', 'force', 'tables']
                       if a.sweeps else ['cases'])
    _os.makedirs(DATA, exist_ok=True)

    if 'cases' in parts:
        print('task rates, 1 s and 10 s')
        arrays, scalars = {}, {}
        for cyc in (FAST, SLOW):
            arr, sc = run_case(cyc)
            tag = TAGS[cyc]
            arrays.update({f'{tag}_{k}': v for k, v in arr.items()})
            scalars[tag] = sc
        np.savez(_os.path.join(DATA, 'cases.npz'), **arrays)
        json.dump(scalars, open(_os.path.join(DATA, 'cases.json'), 'w'), indent=1)
        print('  -> data/cases.npz, data/cases.json')

    if 'gains' in parts:
        print('inhibition-gain sweep (slow)')
        np.savez(_os.path.join(DATA, 'sweep_gains.npz'), **sweep_gains())
        print('  -> data/sweep_gains.npz')

    if 'rates' in parts:
        print('task-rate series')
        json.dump(sweep_rates(), open(_os.path.join(DATA, 'rates.json'), 'w'),
                  indent=1)
        print('  -> data/rates.json')

    if 'force' in parts:
        print('force-level sweep')
        json.dump(sweep_force(),
                  open(_os.path.join(DATA, 'force_sweep.json'), 'w'), indent=1)
        print('  -> data/force_sweep.json')

    if 'tables' in parts:
        print('contributions of the two inhibitory pathways')
        out = {'ablations': ablations()}
        print('spread of the per-unit force gains')
        out['force_gains'] = sweep_force_gains()
        json.dump(out, open(_os.path.join(DATA, 'tables.json'), 'w'), indent=1)
        print('  -> data/tables.json')


if __name__ == '__main__':
    main()
