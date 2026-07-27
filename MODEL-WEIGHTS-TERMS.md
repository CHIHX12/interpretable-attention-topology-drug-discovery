# TEMA-ENM Model Parameters — Terms of Use

**The trained model parameters ("weights") for TEMA-ENM are not distributed in
this repository and are not covered by [`LICENSE.md`](LICENSE.md).** They are
made available on request under the terms below.

This follows the access model used for AlphaFold 3: the source code is open for
inspection and academic reproduction, while the trained parameters are granted
per-request and restricted to noncommercial use.

---

## What is covered

| Artifact | Description | Availability |
|----------|-------------|--------------|
| `DrugBAN_BiLSTM_BindingDB_epoch94.pth` | BiLSTM-BAN base model, pre-trained on BindingDB | On request |
| `DrugBAN_BiLSTM_GHSR_epoch36.pth` | TEMA-ENM, fine-tuned on GHSR (val AUROC 0.9621) | On request |

---

## Permitted use

Access is granted for **noncommercial purposes only**, specifically:

- Academic research, teaching, and study
- Verification and reproduction of results reported in the associated publications
- Method development and benchmarking, where results are published openly

Use by academic institutions, public research organizations, and government
research bodies is noncommercial for the purposes of these terms.

## Prohibited without a separate commercial license

- Use by or on behalf of a for-profit entity for any business purpose
- Incorporation into a commercial product, service, or pipeline
- Providing predictions, analyses, or derived outputs to third parties for a fee
- Redistribution of the parameters, in whole or in part, to any third party
- Training, fine-tuning, or distilling another model from these parameters for
  any purpose that is not itself noncommercial

## Conditions

1. **No redistribution.** You may not publish, share, mirror, or otherwise
   transfer the parameters. Direct others to request access themselves.
2. **Derived parameters inherit these terms.** Any weights produced by
   fine-tuning, distilling, or otherwise deriving from these parameters remain
   subject to this document.
3. **Attribution.** Publications or presentations using these parameters must
   cite both works listed under [Citation](README.md#citation).
4. **No warranty.** The parameters are provided as is, without warranty of any
   kind. They are research artifacts and are **not** validated for clinical,
   diagnostic, or therapeutic use.
5. **Termination.** Access ends immediately on breach of these terms.

---

## Requesting access

Send the following to the corresponding author:

- Name, institutional affiliation, and institutional email address
- A one-paragraph description of the intended use
- Confirmation that the use is noncommercial as defined above

**Chih-Yang Cheng** — Department of Chemistry, National Chung Hsing University
ORCID: https://orcid.org/0009-0002-2694-247X

Requests are reviewed individually. Academic requests are normally granted.

## Commercial licensing

For any use falling outside the permitted scope above, a separate commercial
license is required. Contact the address above. Commercial licensing may require
coordination with National Chung Hsing University's Technology Licensing Office.

## Peer review

Reviewers and editors requiring access to the parameters to assess a submitted
manuscript will be provided a private, time-limited link on request. Please
identify the journal and manuscript.

---

*Last updated: 2026-07-28*
