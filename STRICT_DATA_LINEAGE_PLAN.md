# Strict Data Lineage Plan (Audit-Grade Rebuild)

## Goal

Rebuild the Qwen3 rigorous dataset pipeline so the next training/evaluation cycle is:

1. Technically reproducible
2. Auditable (data lineage + file hashes + split leakage proof)
3. Easy to explain to OSC/auditors

This plan preserves the current model-direction findings (`V2B`/`V2C` IF-priority direction looks reasonable), but upgrades the data governance and evidence quality.

## Scope

Applies to the strict experiment chain only:

1. `data/qwen3_rigorous_train.json`
2. `data/qwen3_rigorous_val.json`
3. `data/qwen3_rigorous_test_labeled.json`
4. `data/qwen3_rigorous_test_if_unlabeled.json`
5. `data/qwen3_rigorous_train_v2b.json` / `v2c` (variant datasets)

## Important Terminology (use this in reports)

1. **Direct experiment inputs**
The files used by training/evaluation commands at runtime (the `qwen3_rigorous*` files).

2. **Upstream lineage sources**
The files used to construct the strict datasets (e.g. `train.json`, `argilla_ifeval.json`, `dataset/IFEval/input_data.jsonl`, etc.).

3. **Atomic source**
A source file with clear provenance (public download/raw source/API-generated batch), not a mixed or derived file.

4. **Derived/mixed source**
A file created from multiple sources where per-sample provenance may be partially lost unless separately tracked.

## Current State (what we know)

1. Confirmed by user:
- `data/train.json` and `data/val.json` are self-generated via API.

2. Current strict pipeline is technically useful but not audit-grade:
- Split isolation and leakage checks exist.
- Source composition is partially visible (`source` labels in rigorous outputs).
- Full upstream provenance is not always preserved at sample level.

3. Current experimental results are still valuable:
- Treat current `V1/V2B/V2C` results as **exploratory evidence** / direction validation.

## Target Deliverables (Audit Pack)

For each strict dataset rebuild, produce:

1. `data/qwen3_rigorous_*.json` (existing outputs)
2. `data/qwen3_rigorous_manifest.json` (existing summary manifest)
3. `data/audit/qwen3_rigorous_lineage.jsonl` (new sample-level lineage)
4. `data/audit/qwen3_rigorous_source_snapshot.json` (new file/source snapshot)
5. `data/audit/qwen3_rigorous_hashes.json` (new SHA256 manifest)
6. `data/source_registry.json` (new source registry; maintained manually + script-assisted checks)
7. `docs/licenses_and_usage.md` (new; recommended)

## What Has Been Implemented (this turn)

`scripts/build_qwen3_rigorous_dataset.py` now supports an audit mode:

1. `--audit-mode`
- Emits lineage, source snapshot, and hash manifests

2. `--audit-dir`
- Custom output directory for audit artifacts (default `data/audit`)

Outputs include:

1. `*_lineage.jsonl`
- Final split membership (`train/val/test_labeled/test_if_unlabeled`)
- source label / source file / source index / dedup key / task metadata

2. `*_source_snapshot.json`
- Direct input files used in that build
- Final split source label distribution
- Final split source file distribution

3. `*_hashes.json`
- SHA256 + size for input files, generated outputs, manifest, and builder script

## Phase 1: Build a Source Registry (must do before audit run)

Create `data/source_registry.json` with one record per upstream source file.

Recommended schema:

```json
{
  "version": "1.0",
  "sources": [
    {
      "path": "data/train.json",
      "source_type": "api_generated",
      "owner": "user",
      "generation_method": "scripts/generate_dataset.py",
      "model_provider": "OpenAI-compatible API",
      "model_name": "REDACTED_OR_ACTUAL",
      "generated_at": "YYYY-MM-DD",
      "license": "internal",
      "sha256": "..."
    }
  ]
}
```

Minimum fields (required):

1. `path`
2. `source_type` (`public_download` / `api_generated` / `derived_mixed` / `raw_prompt_only`)
3. `generation_method` or `download_method`
4. `license`
5. `sha256`

Recommended additional fields:

1. `upstream_dataset`
2. `upstream_revision`
3. `downloaded_at`
4. `notes`

## Phase 2: Rebuild Strict Datasets from Audit-Approved Sources

### Recommended strategy

Prefer atomic, auditable sources. Avoid using opaque derived files as primary inputs if provenance is unclear.

### Preferred upstream set for audit rebuild (suggested)

1. `data/train.json` / `data/val.json` (confirmed self-generated)
2. `data/argilla_ifeval.json` (public IF training)
3. `data/ifeval_full_with_meta.json` (local generated Chinese IF; keep generation evidence)
4. Public translation/summarization source(s) with metadata-preserving output (or regenerated `public_val_v2_with_meta.json`)
5. `dataset/IFEval/input_data.jsonl` (prompt-only IF benchmark source)
6. `dataset/m-ifeval/PMMEval-mifeval-*.json` (multilingual prompt-only IF benchmark source)

### Avoid as primary audit inputs unless provenance is reconstructed

1. `data/train_mixed_3k.json`
2. `data/train_v3.json` / `data/val_v3.json`
3. `data/public_val_v2.json` (unless paired with metadata file or regenerated)
4. `data/ifeval_combined.json`

These can still be used for exploratory experiments, but are not ideal for audit-first strict rebuilds.

## Phase 3: Re-run Training / Validation / Testing (Clean Audit Cycle)

### Minimum rerun matrix (recommended)

1. Base (re-eval only on rebuilt strict test sets)
2. `V1-clean` (rebuilt strict train/val)
3. `V2B-clean` (rebuilt strict train_v2b + same strict val)
4. `V2C-clean` (optional but recommended if you want stable backup comparison)

### Why this is enough

This confirms whether the current direction (`IF` emphasis in `V2B/V2C`) still holds after upgrading data governance.

## Phase 4: Audit-Ready Reporting Package

Prepare one package with:

1. Data lineage report (direct inputs vs upstream sources clearly separated)
2. Source registry + file hashes
3. Strict split manifest (`counts`, `source_distribution`, `leakage`)
4. Training configs and trainer states
5. Evaluation results and scoring script version
6. License/usage notes for public datasets

## Immediate Next Actions (practical)

1. Build `data/source_registry.json` (start with known facts; mark unknowns explicitly)
2. Regenerate/restore metadata-preserving public translation/summarization source file(s)
3. Run `scripts/build_qwen3_rigorous_dataset.py --audit-mode`
4. Inspect `data/audit/*lineage.jsonl` + `*hashes.json`
5. Rebuild `v2b/v2c` variant datasets with provenance retention (next step script/tooling)
6. Start clean rerun (`V1-clean` then `V2B-clean`)

## Reporting Guidance (for auditors)

Use this phrasing:

1. “Current V2B/V2C results established directional validity.”
2. “We then rebuilt the strict dataset pipeline with audit-grade lineage and hash manifests.”
3. “Final claims are based on the audit-grade rerun, not solely on exploratory runs.”

