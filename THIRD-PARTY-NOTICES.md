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
| BindingDB | CC BY 4.0 | Pre-training splits for the BiLSTM-BAN backbone |
| GHSR activity data | See `datasets/README.md` | GPCR fine-tuning |

> **Action required before commercial distribution**: the upstream database for
> the GHSR IC50 activity data is not yet recorded in this repository. Confirm and
> document it. If it derives from ChEMBL (CC BY-SA 3.0), the ShareAlike term
> propagates to redistributed derivative datasets. If it derives from BindingDB
> (CC BY 4.0), attribution alone is sufficient.

---

## Packaged / binary distributions

If TEMA-ENM is redistributed as a compiled executable, container image, or
hosted service, this file must be included in the distribution — for example as
`THIRD-PARTY-NOTICES.txt` alongside the binary, or reachable from an "About" or
"Licenses" screen. This is a condition of the DrugBAN MIT license and is not
optional.
