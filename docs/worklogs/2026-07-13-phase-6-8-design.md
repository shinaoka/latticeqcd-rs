# Phase 6–8 design work log

## Summary

Defined the Phase 6–8 contracts for tenferro extension execution, role-split
autodiff, typed gauge-field storage, SU(3) evolution, and test-only HMC checks.
Added repository rules adapted for gaugefields-rs from tenferro-rs.

## Sources reviewed

- gaugefields-rs Phase 0–5 source, tests, design records, and issues #8–#10;
- tenferro-rs `origin/main` at design-time revision
  `51bc0a7bef274e20d08fc054856cb4d74c284cbe`, including tensor, runtime,
  extension, host-reference, graph executor, typed operation, and AD APIs;
- tenferro-rs `REPOSITORY_RULES.md` at the same revision;
- Gaugefields.jl at `9e5719970770f4497405a856315c90bef7f74449`,
  especially TA exponentiation, NC=3 normalization, and the no-wing HMC test.

## Decisions

- Use `TypedTensor<T>` for owned numerical storage, erased `Tensor` only at the
  extension ABI, and `TracedTensor` only for graph construction.
- Construct and reuse runtime owners explicitly; do not add global registration
  or per-operation backend creation.
- Share Wilson numerical kernels between direct and host-reference paths.
- Implement reverse AD through action linearization followed by JVP transpose;
  do not add a competing direct primal VJP path.
- Preserve the Julia analytic `exp_ta` and fallback algorithm, while delegating
  the batched field matrix product to tenferro `dot_general`.
- Keep HMC orchestration crate-private and test-only.
- Deliver one PR per phase after this design/rules PR.

## Alternatives rejected or deferred

- Static-rank typed tensors: awkward at the dynamic extension ABI and no clear
  benefit beyond wrapper validation.
- Generic eigendecomposition for `exp_ta`: loses reference branch/fallback parity.
- Public HMC sampler and gradient flow: beyond Phase 8's numerical-kernel scope.
- Copying all 853 lines of tenferro-specific repository rules: most govern
  tenferro internals, GPU/FFI, providers, and crate layering not owned here.

## Verification

- Confirmed gaugefields-rs previously had no `REPOSITORY_RULES.md`.
- Read the complete tenferro-rs rules and selected downstream-applicable rules.
- Checked the design and rules with `git diff --check` and searched for unresolved
  `TODO`, `TBD`, and `FIXME` markers.

## Remaining risks

- Implementation must re-fetch tenferro `origin/main` and pin one exact,
  mutually compatible revision because main may advance after this design.
- The public trait spelling used to invoke batched dot may change at that pin;
  ownership, explicit backend, placement, and no-fallback semantics remain fixed.

## Plan review corrections

- Resolved Phase 8 runtime ownership with a public `CpuEvolutionContext` that
  privately owns `CpuBackend` and its associated bounded runtime cache.
- Required cached backend sessions, stable contraction slots `0..3`, observable
  cache stats/clear behavior, persistent same-shape reuse, and transactional
  replacement only after all four direction contractions validate.
- Clarified that `register_runtime` is passed directly to
  `GraphExecutor::register_extension(gaugefields::register_runtime)`.
