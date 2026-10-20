# LatentBridge: Synthetic Support Interventions

Materials for **Marginal-Preserving Support Interventions in Synthetic Transformer-Block Distillation**, Haofeng Hu, Tianjin University, Tianjin, China. Correspondence: 1772027289@qq.com.

**Release candidate 0.1.0-rc1. Prepared for private review; public release is pending explicit author approval. No archival DOI has been assigned.**

## Research question and findings

A complete hidden vector can already contain substantial context. Does rearranging those vectors across positions change what a compressed Transformer block learns? Q_pos independently permutes donor rows at each position, preserving the exact finite multiset of complete position vectors between the Native and Q_pos synthetic banks. Teacher residual-update labels are computed again on the constructed inputs.

| Formal campaign | Teacher / zero-based block | Paired replicates | Registered outcome |
|---|---|---:|---|
| ID2 | GPT-2 / 5 | 48 | Material loss on both endpoints |
| XR1 | GPT-2 / 9 | 24 | Dual-endpoint relative preservation under its 5%-of-Native-recovery rule |
| CMR1 | Pythia-160M / 5 | 48 | Qualified preservation under its absolute margins |

These are three separate registered decisions. Their criteria and ledgers differ. The paper reports the full rules and bounds; this table does not pool them into a new test.

Post-hoc **cross-support** evaluation characterizes the terminal Students on the same independent synthetic inputs. On ID2 Native dev, Native-trained and Q_pos-trained Students have mean NMSE 0.102917 and 0.156757; their ordering reverses on Q_pos dev. Native synthetic support is distinct from the real-language-induced target support used for the formal endpoints. Functional-response decomposition and ID2 short continuation are supplementary exploratory evidence. None is a new registered target endpoint or an identified causal mechanism.

## Contents and quick verification

| Path | Content |
|---|---|
| `paper/` | Signed release-ready manuscript, appendix, bibliography, PDF, portable LaTeX source ZIP and figures |
| `results/formal/{id2,xr1,cmr1}/` | Unchanged persisted aggregate and per-Student metric JSON; no target token arrays |
| `results/cross_support/` | All 120 paired-replicate cross-support records and their summary |
| `results/supplementary/` | Existing feasibility, functional-response and continuation summaries |
| `configs/` | Exact training contract and extracted frozen RNG roles; extraction identities are recorded |
| `src/silent_teacher/` and `scripts/` | Selected original scientific implementation and its local import closure |
| `tools/` | Result verification, figure regeneration, illustrative Q_pos check; no model calls |
| `provenance/` | Original-to-release file identities and full snapshot SHA-256 inventory |

For figure/result checking only (Python 3.12 or later):

```sh
python -m pip install -r requirements-analysis.txt
python tools/verify_release.py
python tools/check_qpos.py
python tools/plot_paper.py
```

Figures are written to `generated/figures/`. The plotter checks manuscript table transcriptions against persisted values, preserves the campaign-specific inferential scales, and uses sample SD for descriptive error bars. Output PDF hashes may differ across plotting environments; the supplied paper figures remain fixed. The checks do not load models, checkpoints, hidden banks, corpora, or target arrays.

For the paper alone, unpack `paper/latent_support_v3_1_source.zip` and compile `main.tex` with Tectonic 0.17.0 (`tectonic -X compile main.tex`) or a suitable TeX distribution with BibTeX. The supplied PDF was checked against an independently compiled ZIP. Pandoc is not needed to compile the supplied TeX.

## Reproduction scope

This snapshot supports result inspection and figure regeneration. It does **not** provide a turnkey bit-identical rerun of every experiment. Historical Student checkpoints, hidden banks, Teacher weights, tokenizer files, language corpora and target arrays are excluded. Target ledgers and internal execution-control manifests are not distributed; hashes referring to them are provenance references, not included files.

The original model/training/evaluation implementations are retained for methodological inspection. Their frozen-identity gates intentionally remain intact, so historical command-line runners will refuse to run without their original manifests and inputs. `configs/` supplies the frozen scientific recipe and seed roles without impersonating those control manifests. See `docs/REPRODUCIBILITY.md` for exact entry points and missing prerequisites. A fresh reconstruction would require independently obtaining licensed dependencies and building a new execution envelope; it must not be represented as the historical formal campaign.

Original aggregate/per-Student JSON and cross-support records retain their original bytes. Historical relative paths inside those records identify the original evidence and do not imply that the named arrays or checkpoints are included. `provenance/source_map.json` maps those identities to the curated snapshot. Historical development commit IDs in the paper identify the research record; that private Git history is not part of this new repository.

## Licenses and responsibility

Original software: **MIT**, see `LICENSE`. Manuscript, figures and authored result documents/JSON: **CC BY 4.0**, see `LICENSE-DOCUMENTS.md`. Third-party packages, weights and corpora retain their own licenses and are not bundled. The affiliation identifies the author and does not imply institutional funding or endorsement.

Generative AI tools, including Codex and other language-model assistants, substantially assisted research planning, programming, analysis and substantive manuscript writing. Haofeng Hu is the sole author and accepts responsibility for verification, scientific claims and the final work. See the paper's full disclosure.

Use `CITATION.cff` for this software snapshot. Cite the paper separately; public repository/DOI identifiers will be added only after a release actually exists. This signed manuscript is not an anonymous TMLR submission.
