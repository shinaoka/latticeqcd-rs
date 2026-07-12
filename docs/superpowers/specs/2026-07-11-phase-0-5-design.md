# Gaugefields-rs Phase 0–5 Design

## Scope

Implement GitHub issues #2 through #7 in one ordered workstream and one final pull request. This establishes the CPU-only SU(3) foundation through the hand-written Wilson gauge force. Phase 6 and later traced execution and automatic differentiation are out of scope.

The issue bodies are normative except for their crates.io dependency pinning proposal. The maintainer's newer decision overrides that proposal: all tenferro dependencies must follow a single exact revision of `tenferro-rs` `origin/main`.

## Dependency policy

Resolve `tenferro-rs` `origin/main` before implementation and pin every Git dependency to the same commit revision. At design approval time that revision is `f504ba0a8668baca89ab1d4348b9475ff85377b4`. Do not use a moving `branch = "main"` dependency in committed manifests. Dependencies that tenferro exposes through its own workspace, including computegraph and tidu where needed, must remain compatible with that revision.

The repository is a standalone Cargo workspace containing one library crate named `gaugefields`. Default features stay minimal. An `autodiff` feature may compile the Phase 0 extension-rule skeleton required by issue #2, but Phase 6–7 behavior is not implemented.

## Ordered batches

### Batch 1: foundation (issues #2 and #3)

Create the workspace, CI, dependency record, layout contract, fixture format, validated lattice/link types, cold SU(3) construction, and fixture reader. The only owned gauge payload is a tenferro `Tensor`; no competing array abstraction is introduced. Errors are typed. Fixtures preserve Julia/Gaugefields.jl column-major layout.

### Batch 2: compute foundation (issues #4 and #5)

Add one dependency-free `Mat3` module using `[Complex64; 9]` in column-major order. All later eager and differentiation code must reuse it. Add x-fastest site indexing, modular periodic neighbors, and direct link access without materialized shift buffers.

### Batch 3: observables and force (issues #6 and #7)

Implement direct plaquette accumulation, normalized plaquette, measurement staple, and Wilson action. Then implement the plaquette-action `dsdu`, projected gauge force, and dense complex action gradient. Keep measurement and force staples distinct. Record the Julia, tenferro Hermitian-inner-product, and TA-basis conventions explicitly.

## Public boundaries

- `LatticeShape4`, `Boundary`, `GaugeLinkTensor`, and `GaugeLinks` own validation and gauge-field structure.
- `Mat3` owns all site-local SU(3) arithmetic used by observables and forces.
- Indexing/link-access helpers own periodic coordinate mapping and tensor block access.
- Observable APIs own plaquette, measurement staple, and Wilson action semantics.
- Force APIs expose three deliberately distinct quantities: `dsdu`, `gauge_force`, and `action_gradient`.
- Public APIs remain no larger than the issue acceptance criteria require.

## Error handling

Invalid lattice extents, dtype, rank, shape, direction, color count, non-Fortran fixtures, and malformed metadata produce typed errors rather than panics. Internal arithmetic may rely on invariants established by validated public constructors. Unsupported `NC != 3` reaches a clear `UnsupportedNc` error before an SU(3)-specific kernel runs.

## Verification

Implementation follows red-green-refactor. Each behavior first gets a focused failing test. Required validation includes:

- formatting, Clippy with warnings denied, and the complete Rust test suite;
- byte/order fixture checks and typed invalid-fixture cases;
- `Mat3` reference and algebraic property tests;
- site/coordinate round trips and all ±direction periodic shifts on a non-isotropic lattice;
- cold-field identities and direct-versus-staple plaquette cross-checks;
- Julia parity for fixtures, observables, `dsdu`, and TA coefficients where the Julia environment is available;
- central finite-difference directional checks demonstrating the documented complex gradient convention and second-order convergence.

Fixture provenance and any unavailable external Julia verification must be reported explicitly; synthetic data must not be represented as Julia-generated parity evidence.

## Delivery

Luna implements the approved batches sequentially in the durable `feat/phase-0-5` worktree. Each phase gets a focused commit after its tests pass. Review first checks issue/spec compliance, then code quality. One final pull request covers all completed issues, and only fully satisfied issues receive `Closes #N` entries.
