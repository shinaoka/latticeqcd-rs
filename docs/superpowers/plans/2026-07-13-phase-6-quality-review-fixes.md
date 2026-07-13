# Phase 6 Code-Quality Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove volume-proportional prepared metadata, strengthen callback numerical oracles, and preserve placement errors across the extension ABI.

**Architecture:** `PreparedGaugeField` stores only four extents and four column-major site strides, computing periodic neighbors with checked O(1) arithmetic. Extension callbacks share one `GaugeError` conversion that unwraps `Placement` sources and otherwise returns family-scoped `InvalidConfig`. Private module tests exercise prepared metadata and callbacks directly; public behavioral tests retain observable parity without source inspection.

**Tech Stack:** Rust, tenferro typed tensors/runtime extensions, checked Julia fixtures, Cargo test/clippy/rustdoc.

---

### Task 1: Constant-size prepared lattice metadata

**Files:**
- Modify: `gaugefields/src/kernel.rs`
- Modify: `gaugefields/tests/observables.rs`

- [ ] Add a private unit regression asserting prepared metadata size is independent of lattice volume and periodic neighbors match `neighbor_site`.
- [ ] Run the focused unit test and record the expected compile failure before the metadata seam exists.
- [ ] Replace neighbor vectors with extents/strides and checked wrap arithmetic, keeping an explicit arithmetic invariant.
- [ ] Remove the brittle source-string test and run shift, observable, force, finite-difference, and Julia parity suites.
- [ ] Commit the focused change.

### Task 2: Nontrivial callback numerical oracles

**Files:**
- Modify: `gaugefields/src/extension/tests.rs`

- [ ] Add random-fixture JVP tests for multiple nonzero active tangents against centered finite differences and the direct gradient inner product.
- [ ] Add force callback tests for seeds `1`, `-2.5`, and `0.25` against every direct-gradient component.
- [ ] Verify RED by temporarily mutating the callback sign/constant output, run the focused tests, record the diagnostic, and restore production immediately.
- [ ] Run the unmodified focused tests green and commit.

### Task 3: ABI error fidelity

**Files:**
- Modify: `gaugefields/src/extension.rs`
- Modify: `gaugefields/src/extension/tests.rs`

- [ ] Add variant-level fake-device tests requiring action-link, JVP-tangent, and force-seed placement failures to remain tenferro placement variants.
- [ ] Run focused tests and record failures from the current stringifying conversions.
- [ ] Centralize conversion so `GaugeError::Placement` returns its source and all domain failures become family-scoped `InvalidConfig`; route tangent and seed host access through it.
- [ ] Run focused tests green and commit.

### Task 4: Documentation and final gate

**Files:**
- Modify: `docs/design/phase-6-8.md`
- Modify: `docs/worklogs/2026-07-13-phase-6.md`

- [ ] Replace bounded-neighbor-table wording with constant-size metadata and document the architectural review rule.
- [ ] Record RED/GREEN evidence, including intentional callback mutations.
- [ ] Run fmt, clippy, both full feature matrices, doctests, rustdoc, and diff hygiene.
- [ ] Commit documentation and report commit SHAs plus command evidence.
