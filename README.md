# gaugefields-rs

CPU-first lattice gauge fields backed by tenferro tensors.

## Compatibility policy

This development line follows `tenferro-rs` `origin/main`, but every Cargo Git
dependency is locked to exact revision
`f504ba0a8668baca89ab1d4348b9475ff85377b4`. Updating to a newer main revision
is an intentional compatibility change: all tenferro pins move together, the
full feature matrix is rebuilt, and fixture/layout tests are rerun.

Until the API stabilizes, compatibility is clean-break rather than emulation:
gaugefields-rs targets its recorded tenferro revision and does not carry shims
for older or newer tenferro APIs.
