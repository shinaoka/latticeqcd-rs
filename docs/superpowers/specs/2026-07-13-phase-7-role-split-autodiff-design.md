# Phase 7 Role-Split Autodiff Design

## Scope and public contract

Phase 7 adds explicit graph-level autodiff for the existing Wilson extension
families. With `autodiff` enabled, `gaugefields::ad_rules()` returns a fresh
`ExtensionRuleSet` containing exactly an action linearize rule and a JVP
linear-transpose rule. Application-owned `AdContext` values own these rule sets.
Rule structs stay private; there is no global registry, eager AD wrapper,
direct primal-VJP rule, force rule, or Phase 8 behavior.

## Forward and reverse graphs

The action linearize rule validates/downcasts its four-input payload and scans
the bounded four-entry tangent list once. With no active tangent it returns
`None`. Otherwise it emits one variable-arity `WilsonActionJvpOp`: four external
primal links followed only by active local tangents. Its role marks links fixed
and tangents active; inactive zero tensors are never emitted.

The JVP linear-transpose rule validates/downcasts the variable-arity JVP. Given
a scalar cotangent, it obtains the four fixed primal links and emits one
`WilsonForceOp` with those links plus the active local seed. Its four outputs
are aligned back only to the JVP tangent slots recorded by `active_dirs`.
Tenferro therefore reaches reverse mode through action linearization followed
by JVP transpose, never a second primal-VJP path.

## Errors, evidence, and complexity

Arity, downcast, fixed-input, mask, tangent, and seed violations return typed
errors without panic. Missing runtime and missing rules remain distinct.
Differentiating force is intentionally unsupported because no force rule is
registered.

Numerical tests use the checked nontrivial `random_2x2x2x2` fixture. JVP subsets
are compared with a centered finite-difference step sweep and
`Re sum(conj(gradient) * tangent)`. Reverse results for every direction are
compared componentwise with `action_gradient` for seeds `1`, `-2.5`, and
`0.25`, including inactive directions. Mutation checks detect sign and
constant-output errors.

Each rule does O(1) work with respect to graph size: it inspects only bounded
local inputs, adds at most one node, and never scans or clones the graph.
Runtime/AD ownership stays outside operation paths. Cross-placeholder symbolic
equality remains limited by
[tenferro-rs #1370](https://github.com/tensor4all/tenferro-rs/issues/1370);
concrete runtime validation still enforces exact equality.
