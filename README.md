# Spinal circuitry as the basis for motor unit-based synergies

A pool of five Hindmarsh-Rose alpha-motoneurons assigned to two MU-modes
(MU1-3 and MU4-5), with five Renshaw cells, two Ib interneurons and Golgi
tendon organ feedback, driven by a sinusoidal descending drive. The question
the model is used to ask is whether spinal circuitry alone makes the two
MU-modes co-vary negatively within a fixed phase of the cycle, while both
still follow the descending drive across phases.

Two files, and a strict split between them: **`model.py` runs the model and
writes its results into `data/`; `figures.py` reads `data/` and draws.**
`figures.py` never runs the model, so redrawing a figure is instant and
nothing about the plotting can change a result.

```
model.py      the model, the analyses, and the runs that fill data/
figures.py    every figure, drawn from data/ alone
data/         the stored results
figures/      the figures, as committed
```

Renshaw connectivity is complete: every Renshaw cell inhibits every alpha-MN
including its own, and no gain is zero. The modes are defined by the ratio of
the gains (`wR_cross / wR_within` = 10), not by which connections exist.

## Running

Needs `numpy`, `scipy`, `matplotlib` and `numba`. The integrator is
`@njit`-compiled, so the first run pays a compile cost.

```
python model.py                # the two task rates -> data/cases.{npz,json}
python model.py --sweeps       # everything, including the three slow sweeps
python model.py --only rates   # just one part
python figures.py              # all eight figures from data/
python figures.py 2 3 6        # only those
```

`python model.py` alone takes about a minute and is enough for figures 1, 2,
3, 4 and 6. The three sweeps behind figures 5, 7 and 8 take tens of minutes,
which is why their results are committed in `data/` and `--sweeps` is opt-in.
If a figure needs a file that is not there, `figures.py` says which command
produces it.

### What model.py writes

| file | produced by | feeds |
|---|---|---|
| `data/cases.npz` | default run | figures 1, 2, 3, 4, 6 |
| `data/cases.json` | default run | the scalar indices quoted in the text |
| `data/sweep_gains.npz` | `--sweeps` | figure 5 |
| `data/rates.json` | `--sweeps` | figure 7 |
| `data/force_sweep.json` | `--sweeps` | figure 8 |
| `data/tables.json` | `--sweeps` | the ablation and force-gain tables |

### The figures

| figure | file | content |
|---|---|---|
| 1 | `pca_modes.png` | MU-modes identified by PCA with varimax rotation |
| 2 | `fig2_bars_both.png` | mode firing frequency per phase window, both rates |
| 3 | `fig3_scatter_both.png` | mode co-variation within a window, both rates |
| 4 | `fig4_cellwise.png` | unit-by-unit correlations, both rates |
| 5 | `fig5_sweep.png` | the 2-D inhibition-gain sweep |
| 6 | `fig6_ucm.png` | UCM decomposition, both rates |
| 7 | `fig7_rates.png` | indices against task rate |
| 8 | `force_level.png` | indices against force level |

## Using the model directly

```python
import model as M

spk, F, I, cns, p = M.run(**M.cfg(1000.0))   # 1 s cycle
res = M.analyse(spk, F, cns, p)              # the two correlation analyses
u   = M.ucm_indices(spk, F, cns, p)          # V_UCM / V_ORT / dV

res['r_across_bins']   # +0.97  modes follow the drive across phase windows
res['r'].mean()        # -0.79  modes oppose each other within a window
u['dV'].mean()         # +1.54  synergy index, bounded +/-2
```

`M.cfg(cyc)` holds neural constants fixed across task rates while scaling the
task-related ones (noise correlation time, sharing fluctuation time, smoothing
window) with cycle duration, as they would be in an experiment.

## Key parameters

In the `DEF` dict in `model.py`:

| name | value | meaning |
|---|---|---|
| `tau_hr` | 7.4 ms | time scaling of the HR equations; sets ~12 Hz firing |
| `tau_syn_ms` | 100 | inhibitory synaptic decay |
| `tau_act_ms` | 100 | twitch / activation time constant |
| `wR_self`, `wR_within`, `wR_cross` | 0.3, 0.3, 3.0 | Renshaw gains |
| `w_Ib` | 0.5 | Ib interneuron inhibition |
| `delay_rc_ms`, `delay_ib_ms` | 5, 50 | conduction delays |
| `hann_ms` | 1000 | smoothing window for firing frequency |
| `period_cv`, `amp_cv`, `gain_cv` | 0.05, 0.05, 0.10 | parameter variability |
| `force_gain` | 0.9 to 1.2 | per-unit force at full activation |
| `gto_mode` | `total` | Golgi senses total pool force; `per_mode` is the alternative |

## Known loose ends

- `data/force_sweep.json` is from a sibling run of the force sweep, not the
  exact run behind `figures/force_level.png`. The shape is the same but the
  peak of dV reads +1.60 from the cache against +1.65 in the committed
  figure. `python model.py --only force` then `python figures.py 8` makes the
  two consistent.
- `force_gain` spans 0.9 to 1.2, a ratio of 1.33, which has no literature
  source. The whole-pool figure in Fuglevand, Winter & Patla (1993) is a
  100-fold range, but that covers every unit in the muscle rather than five
  units firing together at one force level. `python model.py --only tables` widens
  the spread to 13-fold: r moves by 0.03 and dV by 0.16.
- The pool has five units against the ten to twenty-seven accepted in the
  experimental recordings; scaling with pool size is untested.
