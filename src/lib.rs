#![cfg_attr(feature = "fail-on-warnings", deny(warnings))]

pub mod arbitrary;
mod control_flow;
mod data;
mod exec_state;
mod execution;
mod memory;
mod metrics;
mod value;

pub use self::{
	data::MAX_INSTR_INPUTS, exec_state::*, execution::*, memory::*, metrics::*, value::*,
};
