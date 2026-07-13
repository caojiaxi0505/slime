# HF Metadata Normalize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load-time normalize official HF SWE rows into flat Path A `Sample.metadata`, including DataEng-compatible image URLs with a configurable TCR registry prefix.

**Architecture:** New `examples/claudecode_ags/dataset_normalize.py` owns `normalize_official_row` / `resolve_image`. `generate._parse_metadata` calls it when `dataset_type` is set. No `slime/` core changes; no offline converter.

**Tech Stack:** Pure Python; pytest; env `SLIME_CC_IMAGE_REGISTRY`.

**Spec:** [2026-07-10-hf-metadata-normalize-design.md](../specs/2026-07-10-hf-metadata-normalize-design.md)

---

## File map

| File | Responsibility |
|------|----------------|
| `examples/claudecode_ags/dataset_normalize.py` | type aliases, registry, image formulas, field mapping |
| `examples/claudecode_ags/generate.py` | call normalize when `dataset_type` present |
| `tests/claudecode_ags/test_dataset_normalize.py` | image truth table + field pass-through |
| `examples/claudecode_ags/CHECKLIST.md` | mark adapter done |

---

### Task 1: `dataset_normalize` + unit tests

**Files:**
- Create: `examples/claudecode_ags/dataset_normalize.py`
- Test: `tests/claudecode_ags/test_dataset_normalize.py`

- [ ] **Step 1:** Implement `canonical_dataset_type`, `image_registry`, `resolve_image`, `normalize_official_row` per spec.
- [ ] **Step 2:** Tests for all six types, custom registry, preserve image, data_source, ValueError.
- [ ] **Step 3:** `pytest tests/claudecode_ags/test_dataset_normalize.py -v` green.

### Task 2: Wire `generate._parse_metadata`

**Files:**
- Modify: `examples/claudecode_ags/generate.py`
- Test: extend `test_dataset_normalize.py` or small generate parse test

- [ ] **Step 1:** If `metadata.dataset_type` non-empty → `normalize_official_row` first, then existing parsing.
- [ ] **Step 2:** Assert `_parse_metadata` fills `image` for swebench row without pre-set image.
- [ ] **Step 3:** Full `tests/claudecode_ags/` regression; update CHECKLIST.

---

## Success criteria

1. Six `dataset_type`s produce correct TCR images by default.
2. `SLIME_CC_IMAGE_REGISTRY` overrides prefix only.
3. Rows without `dataset_type` unchanged.
4. Suite green.
