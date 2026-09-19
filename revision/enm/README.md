# Elastic-network analysis

Sections 2.7 and 3.6 of the main text, and Supplementary Information 5.5.
These scripts use experimental coordinates only. No machine-learning output
enters any of them.

Each script resolves its inputs relative to its own location, so run them from
this directory:

```bash
cd revision/enm
python anm_tm6_direction.py
```

## Inputs

| File | What it is |
|---|---|
| `6ko5.pdb` | antagonist-bound GHSR, the resting reference for the network |
| `8jsr.pdb` | agonist-bound GHSR (anamorelin) |
| `7f83.pdb` | inverse-agonist-bound GHSR |
| `_data_compact.json` | the contact network built from 6KO5: 293 nodes, 1,590 edges, `k = 1/d` for minimum heavy-atom distances below 5 Å |
| `_anm_cache.npz` | intermediate ANM modes, rewritten by `anm_tm6_direction.py` |

## Scripts and what they report

| Script | Output | Values it reproduces |
|---|---|---|
| `anm_tm6_direction.py` | `FigS2` | crystal transition of TM6 (intracellular +3.81 Å, extracellular +0.05 Å on activation; +0.25 and +3.27 Å for the inverse agonist), least-displaced TM6 residue Phe286 at 0.47 Å against Trp276 at 2.23 Å, and the ANM overlap with the transition (0.72 over ten modes, 0.80 over twenty) |
| `prs_tm6.py` | `FigS5` | perturbation response scanning, force at Phe272 directed at Pro224: the intracellular segment responds inward, −0.517 in radial units |
| `w276_strike_pif.py` | `FigS4` | strike at Trp276 and the response at the PIF and CWxP positions |
| `pif_cwxp_compare.py` | `FigS3` | PIF distances and Trp276 / Phe272 rotamers across the three states |
| `test_step_robustness.py` | `FigS1` | parameter sensitivity: γ from 0.01 to 0.30, dt ≤ 0.01, 100 to 2,000 steps |
| `reverse.py` | printed | the blind reverse-derivation control: ranking all 293 residues by hub-free structural metrics does not recover the attention-selected residues |
| `full_audit.py` | printed | re-checks every numerical value reported in the elastic-network section |
| `enm_demo.py` | printed | the minimal damped-harmonic strike, for reading the method rather than for the paper |

`ENM_WAVE_METHODS.md` and `GHSR_WAVE_METHODOLOGY.md` are the working notes for
the method; `VERIFICATION_REPORT.md` is the audit record.

## Two results that did not survive

Both are reported in the manuscript rather than dropped.

- Trp276 was described as the pivot of TM6 rotation in the previous version. In
  the superposition the least-displaced TM6 residue on activation is Phe286, so
  Trp276 is now reported only as the residue with the highest cross-helix
  restraint.
- The perturbation response test of whether Phe272 engaging Pro224 drives the
  outward swing returns an inward response of the intracellular segment. The
  negative result is reported.
