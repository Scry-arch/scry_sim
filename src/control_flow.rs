use crate::{
	data::{QueueCall, QueueRead},
	memory::{Address, MemoryInstr, MemoryRead},
	report::{Report, Reporter},
};
use scry_isa::Instruction;
use std::collections::VecDeque;

enum CallType
{
	Call,
	Portal,
}

struct CFLData<A: Address>
{
	call_stack: VecDeque<(
		// Return address (when call returns, which address to begin execution at
		A,
		CallType,
		// Branch Stack for that function
		VecDeque<(
			// Location (when to branch)
			A,
			// Target (where to branch to)
			// If none, its a return, otherwise a normal branch
			Option<A>,
		)>,
	)>,
}
struct ReportData
{
	instr: Option<Instruction>,
}

/// Control-Flow (CFL) manager.
/// Also manages the decoding of instructions
pub struct CFLDecode<A: Address>
{
	data: CFLData<A>,
	report: ReportData,
}
impl<A: Address> CFLDecode<A>
{
	fn decode(
		self,
		mem: MemoryInstr<A>,
		queue: QueueCall,
	) -> (Instruction, CFLIssue<A>, MemoryRead<A>, QueueRead)
	{
		todo!()
	}

	fn skip(self, mem: MemoryInstr<A>, queue: QueueCall)
		-> (CFLIssue<A>, MemoryRead<A>, QueueRead)
	{
		(
			CFLIssue {
				data: self.data,
				report: self.report,
			},
			mem.skip(),
			queue.skip(),
		)
	}
}
impl<A: Address> Reporter for CFLDecode<A>
{
	type Args = ();
	type Report = CFLReport<A>;

	fn step(self, _: Self::Args) -> Self::Report
	{
		CFLReport {
			data: self.data,
			report: self.report,
		}
	}
}

pub struct CFLIssue<A: Address>
{
	data: CFLData<A>,
	report: ReportData,
}
impl<A: Address> CFLIssue<A>
{
	fn branch(self, location: A, target: Option<A>) -> CFLReport<A>
	{
		todo!()
	}

	fn call(self, location: A, typ: CallType) -> CFLReport<A>
	{
		todo!()
	}
}
impl<A: Address> Reporter for CFLIssue<A>
{
	type Args = ();
	type Report = CFLReport<A>;

	fn step(self, _: Self::Args) -> Self::Report
	{
		CFLReport {
			data: self.data,
			report: self.report,
		}
	}
}

pub struct CFLReport<A: Address>
{
	data: CFLData<A>,
	report: ReportData,
}
impl<A: Address> Report for CFLReport<A>
{
	type Reporter = CFLDecode<A>;

	fn reset(self) -> Self::Reporter
	{
		CFLDecode {
			data: self.data,
			report: self.report,
		}
	}
}
