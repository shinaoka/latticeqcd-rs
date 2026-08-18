//! Signed Wilson paths and general closed loop actions.
//!
//! The path convention and helper ordering follow the MIT-licensed
//! [Wilsonloop.jl](https://github.com/akio-tomiya/Wilsonloop.jl) 0.1.5 source at
//! revision `e1a617fdedb19b785f89bdeb13c30e53b20743a7`. Gauge storage and the
//! fixed-size SU(3) algebra remain owned by [`gaugefields`].

mod action;
mod error;
mod evaluate;
mod path;

pub use action::{LoopAction, LoopTerm};
pub use error::WilsonError;
pub use evaluate::{evaluate_path, loop_action_force, loop_action_value, loop_trace_sum};
pub use path::WilsonPath;

#[cfg(test)]
mod tests;
