# Data and Figures License

This project distributes material from more than one source, under more than one
set of terms. Read the table before reusing anything.

| Material | License | Why |
|----------|---------|-----|
| `datasets/GPCR_resarch/*.csv` (GHSR activity data) | **CC BY-SA 4.0** | Derived in part from ChEMBL (CC BY-SA 3.0). ShareAlike propagates. |
| `datasets/bindingdb/*.csv` | **CC BY 4.0** | From BindingDB, redistributed via DrugBAN. |
| `datasets/biosnap/`, `datasets/human/` | See upstream | Redistributed via DrugBAN; consult the original sources. |
| Figures and analysis outputs (`Important_Analysis/`, `result/`) | **CC BY-NC 4.0** | Our own creative work, generated from model outputs. |
| Source code | PolyForm Noncommercial 1.0.0 | See [`LICENSE.md`](LICENSE.md) |
| Model parameters | **CC BY-NC 4.0**, open deposit at [doi:10.5281/zenodo.22841754](https://doi.org/10.5281/zenodo.22841754) | See [`MODEL-WEIGHTS-TERMS.md`](MODEL-WEIGHTS-TERMS.md) |

---

## GHSR activity data — CC BY-SA 4.0

<https://creativecommons.org/licenses/by-sa/4.0/>

The GHSR training set is assembled from **ChEMBL** (CC BY-SA 3.0) and **BindingDB**
(CC BY 4.0), with an added `cnnscore` column computed by us via AutoDock-GPU.

Because ChEMBL is licensed **ShareAlike**, any adapted version of this dataset that
you distribute must itself be released under CC BY-SA 4.0 (or CC BY-SA 3.0).
**A NonCommercial restriction cannot be applied to this material** — CC BY-NC is
not a compatible license for ShareAlike content.

You must attribute ChEMBL and BindingDB. Suggested form:

> Contains data from ChEMBL (EMBL-EBI), licensed under CC BY-SA 3.0, and from
> BindingDB, licensed under CC BY 4.0. Modified for this work.

**Note**: ShareAlike constrains redistribution of the *dataset*. It does not, on
the prevailing interpretation, extend to trained model parameters, which are
governed by [`MODEL-WEIGHTS-TERMS.md`](MODEL-WEIGHTS-TERMS.md). This question is
not settled in case law; parties with commercial exposure should take their own
advice.

## Figures and analysis outputs — CC BY-NC 4.0

<https://creativecommons.org/licenses/by-nc/4.0/>

Figures produced by the analysis scripts, and the curated outputs under
`Important_Analysis/`, are our own work and are released under CC BY-NC 4.0:
share and adapt with attribution, noncommercial use only.

Attribution means citing the two works listed under
[Citation](README.md#citation).

---

## Commercial use

The CC BY-SA 4.0 material above may be used commercially, provided the ShareAlike
and attribution terms are honoured. Everything else — the source code, the model
parameters, and the figures — requires a separate commercial license.

Contact **Chih-Yang Cheng** (ORCID: https://orcid.org/0009-0002-2694-247X),
Department of Chemistry, National Chung Hsing University.
