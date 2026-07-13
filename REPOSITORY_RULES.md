# Repository Rules

These rules apply on top of the shared tensor4all rules. They are adapted from
`tenferro-rs/REPOSITORY_RULES.md` for this downstream gauge-field library; rules
about tenferro's own internal crates, GPU/FFI implementation, and provider
selection are intentionally not copied.

## Public Surface

- Keep the public API intentionally small. Prefer `pub(crate)` for runtime glue,
  validation metadata, AD operation constructors, reference kernels, test
  drivers, and other implementation details.
- Public items are deliberate user-facing contracts. Public types implement
  `Debug`, using a compact hand-written implementation when deriving it would
  expose or dump tensor storage.
- README, rustdoc, examples, and design documents must not claim capabilities
  beyond the current public surface. Check them whenever that surface changes.
- Use `tensor4all`, not `Tensor4all`, in project prose unless preserving a proper
  noun or quotation.
- Fallible public operations return the crate's typed `Result`; do not retain a
  panicking compatibility shim unless maintainers explicitly require one.

## Public Boundary Safety

- Validate input-derived dtype, rank, shape, lattice extents, direction, and
  operation payload before shortcuts, allocation, graph emission, or execution.
- Use checked arithmetic for shape products, byte lengths, strides, offsets,
  and allocation sizes before integer conversion.
- Invalid public input must not reach `panic`, `unwrap`, `expect`, unchecked
  indexing, or debug-only assertions. Return a typed error.
- Put repeated eager, traced, and extension validation in shared helpers that
  return prepared metadata. Do not duplicate shape and dtype logic across
  operation surfaces.
- A performance-sensitive non-obvious assumption must have a nearby
  `// INVARIANT: <reason>` marker. Keep `// SAFETY:` for unsafe blocks.
- Keep algorithm, graph, extension, and AD layers free of `unsafe`. A future
  backend/FFI leaf is the only acceptable owner of required unsafe code.

## Tensor And Runtime Boundaries

- Use `TypedTensor<T>` or typed borrowed views for validated internal numeric
  storage and kernels. Use dynamic `Tensor` only at dtype-erased tenferro ABI
  boundaries, and `TracedTensor` only for graph construction.
- Owned tensors follow tenferro's compact column-major layout. Color/contraction
  dimensions precede lattice batch dimensions.
- Do not introduce implicit CPU/GPU transfers. Host-only direct kernels reject
  backend buffers with a typed placement error; callers explicitly transfer.
- Construct backends, executors, AD contexts, and their caches once per
  execution owner and reuse them. Do not hide them in process-global or
  thread-local state and do not construct them inside operation calls.
- Extension runtimes and AD rules are registered explicitly. Missing
  registration is an error; never silently fall back to an eager/reference CPU
  path or a newly constructed backend.
- Use existing tenferro tensor/backend operations for contractions, reductions,
  and dense linear algebra when they express the operation. A custom gauge
  kernel needs a documented reason such as periodic stencil semantics, fixed
  SU(3) algebra, or exact reference-algorithm parity.
- Avoid hidden tensor-sized materialization and CPU/GPU copies. When required by
  an ABI or compact-output contract, make the boundary explicit and test it.

## Extension And AD Rules

- Each extension family owns stable family/version metadata, shared validation,
  shape inference, a host reference runtime, and explicit registration.
- Extension AD rules belong to an explicit `tenferro_ad::AdContext` or rule set,
  never a process-global registry.
- Follow tenferro's graph-level role split: `linearize` emits the JVP operation;
  transposing that linear operation emits the force/VJP operation. Do not add a
  second value/tape-level AD model.
- Every supported AD rule requires numerical oracle coverage. Prefer an
  independent Julia reference plus finite differences; at minimum require
  finite differences and manifest/registration coverage.
- Unsupported higher-order differentiation returns an intentional typed error.

## Performance

- Validate once, borrow tensor storage once, and carry prepared lattice/stride
  metadata into hot loops. Do not repeatedly materialize or request host slices
  per site or link.
- Avoid heap allocation, graph planning, backend construction, and repeated
  full-shape hashing inside hot loops. Reuse scratch and precomputed neighbor
  tables where appropriate.
- Use tenferro `dot_general` for batched matrix multiplication rather than a
  downstream naive loop. Dedicated fixed-size `Mat3` arithmetic and periodic
  stencils are permitted when their rationale is recorded in the design.
- Performance claims require release-mode measurements over representative
  lattice sizes. Numerical tests report a useful maximum or norm residual.

## Tests And Documentation

- Public behavior belongs in integration tests. Keep production files focused;
  prefer module-local `src/<module>/tests/*.rs` for private unit tests, leaving
  only `#[cfg(test)] mod tests;` in the production module.
- Julia-derived fixtures record the upstream file URL, revision, parameter
  choices, layout conversion, and tolerance. Generated values are not a
  substitute for independent finite-difference tests.
- Non-trivial public examples must compile and run in CI as doctests or checked
  examples; do not use `ignore` or `no_run` to hide drift.
- Durable architectural decisions live under `docs/design/`. Non-trivial
  implementation PRs also leave a concise reviewer-facing record under
  `docs/worklogs/` containing sources consulted, decisions, verification, and
  remaining risks. Work logs are not raw transcripts.
- Before completion, run formatting, linting, unit/integration tests, doctests,
  and any relevant source-contract checks. Record exact verification commands
  in the PR work log.
