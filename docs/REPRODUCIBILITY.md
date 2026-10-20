# Reproduction and implementation map

## What can be checked with this snapshot

1. `tools/verify_release.py` verifies the complete release inventory, all three formal aggregate-result identities, campaign replicate counts and all 120 common-support records. It recomputes descriptive four-cell means/SDs and paired differences and compares them with the archived summary.
2. `tools/plot_paper.py` regenerates the four paper figures from existing JSON and checks table transcriptions and all 18 bibliography keys. Figure 2 keeps ID2/CMR1 D bounds separate from XR1 Z bounds. It introduces no new inferential tests.
3. `tools/check_qpos.py` demonstrates the finite complete-vector marginal invariant on a tiny deterministic array. This is an indexing illustration; the production PyTorch implementation and frozen generators are in `h9_id2.py` and `cmr1_campaign.py`.
4. The source ZIP independently compiles to the signed manuscript and appendices. No model inference is required.

## Scientific implementation

| Component | Original implementation retained in snapshot |
|---|---|
| GPT-2 residual-update Teacher and compressed Student | `src/silent_teacher/full_block.py` |
| GPT-2 paired training loop and seeded initialization | `src/silent_teacher/h9_id1.py:train_paired_student`, `seeded_student` |
| Q_pos construction, per-position permutations, invariant checks, training contract | `src/silent_teacher/h9_id2.py` |
| General full-block trainer and fixed schedule | `src/silent_teacher/full_block_trainer.py` |
| ID2 / XR1 recipes and endpoint decisions | `h9_id2.py`, `h9_id2_analysis.py`, `h9_xr1.py`, `h9_xr1_analysis.py` |
| Pythia parallel block / pinned Teacher wrapper | `cmr1_student.py`, `cmr1_teacher.py` |
| CMR1 banks, training and formal arithmetic | `cmr1_campaign.py:build_banks,train_arm`, `cmr1_evaluation.py` |
| Historical terminal-Student common-support scoring | `scripts/run_historical_cross_support.py` |
| Original synthetic summary arithmetic | `scripts/summarize_historical_cross_support.py` |

The original scripts preserve their historical path conventions and identity checks. They are reference implementations, not commands validated as runnable in this curated repository. The local source import closure is included; a successful static import/parse check does not establish numerical reproduction. The release preparation ran no model forward passes or optimizer updates.

## Missing prerequisites for model reruns

- Locally acquired, hash-verified GPT-2 and Pythia-160M weights and matching tokenizers. The manuscript records Pythia revision `50f5173d932e8e61f858120bcb800b97af589f46`; source hashes identify the original GPT-2 and Pythia weights.
- Historical target-selection ledgers, token arrays and predecessor-exposure records. Aggregate metrics and content hashes are included, but these do not reconstruct the selected text windows.
- Historical checkpoints for evaluating the exact trained Students. Their hashes are retained in metric records; checkpoint payloads are excluded.
- The original execution-control manifests and immutable Git tags used by the guarded launchers. The exported `configs/*_rng_roles.json` files retain all scientific campaign role seeds and original manifest SHA-256 identities, but are deliberately not substituted for the missing control files.
- Compatible GPU execution conditions and deterministic numerical libraries. `requirements-lock.txt` is the original environment lock, while `requirements-analysis.txt` is the smaller static plotting environment. Platform and CUDA constraints remain relevant to numerical identity.

The common scientific recipe is 10,000 terminal optimizer updates; microbatch 8, accumulation 4; AdamW (0.9,0.999), epsilon 1e-8, zero weight decay; peak LR 1e-3, 200-step warmup, cosine decay to 1e-4; gradient clipping 1.0; fp32, AMP/TF32 disabled, deterministic algorithms; train/dev/audit counts 4096/256/512; length 128 and residual width 768. The extracted contract preserves additional fields from the original source. Student geometry and architecture-specific details remain in the manuscript and source.

## Data and interpretation

`results/formal` contains numerical evaluation records, including real-language-induced target metrics. It contains no target text, token-ID arrays or third-party datasets. The formal Context endpoint includes constructed history-replacement inputs; it is a functional fidelity measure rather than an ordinary semantic-comprehension task. Independent synthetic supports are separate objects.

Cross-support uses one Teacher-label energy denominator per support column. Functional-response decomposition includes fixed-common-denominator comparisons. These normalizations cannot be interchanged. Short continuation reports all prespecified times; its audit is exploratory and does not select a checkpoint. See manuscript Appendix D/E for complete definitions.
