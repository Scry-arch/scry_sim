use crate::{
	control_flow::{CFLDecode, CFLIssue, CFLReport},
	data::{QueueRead, QueueReport},
	memory::{Address, MemoryRead, MemoryReport},
	report::{Report, Reporter},
};
use scry_isa::Instruction;
use std::marker::PhantomData;

/// Tracks the state of the execution.
struct ExecState<A: Address>
{
	phantom: PhantomData<A>,
}

/// Reports what happened after execution
struct ExecReport<A: Address>
{
	state: ExecState<A>,
}

impl<A: Address> Reporter for ExecState<A>
{
	type Args = (Instruction, MemoryRead<A>, CFLIssue<A>, QueueRead);
	type Report = (ExecReport<A>, MemoryReport<A>, CFLReport<A>, QueueReport);

	fn step(self, args: Self::Args) -> Self::Report
	{
		todo!()
	}
}

impl<A: Address> Report for ExecReport<A>
{
	type Reporter = ExecState<A>;

	fn reset(self) -> Self::Reporter
	{
		ExecState {
			phantom: PhantomData,
		}
	}
}
