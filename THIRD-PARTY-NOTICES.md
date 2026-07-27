# Third-Party Notices

TEMA-ENM incorporates or builds upon the third-party components listed below.
Each remains subject to its own license. These notices must be reproduced in
any distribution of TEMA-ENM, including binary or packaged distributions.

---

## DrugBAN

Portions of the training scaffold in this repository — `ban.py`, and parts of
`models.py`, `trainer.py`, `configs.py`, and `dataloader.py` — are derived from
[DrugBAN](https://github.com/peizhenbai/DrugBAN).

> MIT License
>
> Copyright (c) 2022 peizhenbai
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

**Note on relicensing**: The MIT License permits sublicensing. TEMA-ENM as a
combined work is distributed under the PolyForm Noncommercial License 1.0.0
(see [`LICENSE.md`](LICENSE.md)), with the above notice retained as required.
This does not restrict anyone's independent rights to obtain the original
DrugBAN under its own MIT terms.

---

## Runtime dependencies

TEMA-ENM depends on the following packages at runtime. None impose copyleft
obligations on TEMA-ENM or on products built with it.

| Package | License |
|---------|---------|
| PyTorch (`torch`, `torchvision`, `torchaudio`) | BSD-3-Clause |
| DGL (`dgl`) | Apache-2.0 |
| DGL-LifeSci (`dgllife`) | Apache-2.0 |
| RDKit (`rdkit`) | BSD-3-Clause |
| SELFIES (`selfies`) | Apache-2.0 |
| NumPy (`numpy`) | BSD-3-Clause |
| SciPy (`scipy`) | BSD-3-Clause |
| pandas (`pandas`) | BSD-3-Clause |
| scikit-learn (`scikit-learn`) | BSD-3-Clause |
| yacs (`yacs`) | Apache-2.0 |
| PrettyTable (`prettytable`) | BSD-3-Clause |
| tqdm (`tqdm`) | MPL-2.0 AND MIT |
| Matplotlib (`matplotlib`) | PSF-based (matplotlib license) |
| seaborn (`seaborn`) | BSD-3-Clause |

`tqdm` is dual-licensed MPL-2.0 and MIT. MPL-2.0 is file-level copyleft and
imposes no obligation on separate works that merely depend on it.

---

## Data sources

| Source | Terms | Used for |
|--------|-------|----------|
| [ChEMBL](https://www.ebi.ac.uk/chembl/) (EMBL-EBI) | **CC BY-SA 3.0** | GHSR IC50 activity data (fine-tuning) |
| [BindingDB](https://www.bindingdb.org/) | CC BY 4.0 | GHSR activity data; pre-training splits for the BiLSTM-BAN backbone |
| AutoDock-GPU (computed by us) | This project's terms | `cnnscore` docking column |

### Required attribution

Any redistribution of the GHSR dataset must carry:

> Contains data from ChEMBL (EMBL-EBI), licensed under CC BY-SA 3.0, and from
> BindingDB, licensed under CC BY 4.0. Modified for this work.

### ShareAlike — scope and limits

ChEMBL is licensed **ShareAlike**. Consequences:

1. **The GHSR dataset is distributed under CC BY-SA 4.0**, not CC BY-NC 4.0.
   A NonCommercial restriction cannot lawfully be applied to ShareAlike material.
   See [`LICENSE-DATA.md`](LICENSE-DATA.md).
2. **ShareAlike does not extend to the source code**, which is an independent work
   under PolyForm Noncommercial 1.0.0.
3. **ShareAlike is not treated as extending to trained model parameters.** Whether
   model weights constitute a derivative work of their training data is unsettled;
   the prevailing interpretation, relied on across the industry, is that they do
   not. Parties with commercial exposure should take their own advice.

### Note for commercial distribution

If a commercial product ships the GHSR dataset, the CC BY-SA 4.0 terms and the
attribution above travel with it. A product that ships only the trained
parameters, or that serves predictions without distributing the dataset, does not
trigger ShareAlike on the dataset.

---

## Packaged / binary distributions

If TEMA-ENM is redistributed as a compiled executable, container image, or
hosted service, this file must be included in the distribution — for example as
`THIRD-PARTY-NOTICES.txt` alongside the binary, or reachable from an "About" or
"Licenses" screen. This is a condition of the DrugBAN MIT license and is not
optional.
