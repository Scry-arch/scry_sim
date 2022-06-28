use crate::data::OperandQueue;
use std::collections::{HashMap, VecDeque};

enum ControlFlowType
{
	Branch(usize),
	Call(usize),
	Return,
}

struct CallFrame
{
	/// Where execution should continue after a return instruction
	ret_addr: usize,

	/// Issued branches that have yet to trigger,
	/// Key is the address after which the branch should trigger
	branches: HashMap<
		// Location (when to branch)
		usize,
		// Branch type including potential target
		ControlFlowType,
	>,
}

/// Controls the execution of the program.
pub struct ControlFlow
{
	/// Previously returned address + 2.
	/// If no control flow is issued, then this is the next address to be
	/// returned. If not, the control flow decides the next address to be
	/// issued.
	next_addr: usize,
	call_frame: CallFrame,
	call_stack: VecDeque<CallFrame>,
	report: ReportData,
}
struct ReportData
{
	issued_branches: usize,
	issued_calls: usize,
	issued_returns: usize,
	triggered_branches: usize,
	triggered_calls: usize,
	triggered_returns: usize,
}
impl ReportData
{
	fn new() -> Self
	{
		Self {
			issued_branches: 0,
			issued_calls: 0,
			issued_returns: 0,
			triggered_branches: 0,
			triggered_calls: 0,
			triggered_returns: 0,
		}
	}
}

impl ControlFlow
{
	pub fn new(start_addr: usize) -> Self
	{
		Self {
			next_addr: start_addr,
			call_frame: CallFrame {
				ret_addr: start_addr,
				branches: HashMap::new(),
			},
			call_stack: VecDeque::new(),
			report: ReportData::new(),
		}
	}

	/// Returns the address of the next instruction to execute.
	///
	/// The given queue is suitably updated if a call or return is triggered
	/// before the returned address.
	///
	/// If the call stack is empty after a return is triggered, None is
	/// returned.
	pub fn next_addr(&mut self, queue: &mut OperandQueue) -> Option<usize>
	{
		if let Some(cft) = self
			.call_frame
			.branches
			.remove(&self.next_addr.saturating_sub(2))
		{
			use ControlFlowType::*;
			match cft
			{
				Branch(tar) =>
				{
					self.next_addr = tar + 2;
					self.report.triggered_branches += 1;
					Some(tar)
				},
				Call(tar) =>
				{
					let old_next = self.next_addr;
					self.next_addr = tar + 2;
					queue.push_queue();
					let old_frame = std::mem::replace(
						&mut self.call_frame,
						CallFrame {
							ret_addr: old_next,
							branches: HashMap::new(),
						},
					);
					self.call_stack.push_back(old_frame);
					self.report.triggered_calls += 1;
					println!("Trigger call: {},{},{}", old_next, self.next_addr, tar);
					Some(tar)
				},
				Return =>
				{
					let ret_addr = self.call_frame.ret_addr;
					self.next_addr = ret_addr + 2;

					// Inputs to retur

					queue.pop_queue();
					self.call_frame = self.call_stack.pop_back()?;
					self.report.triggered_returns += 1;
					println!("Trigger ret: {},{},{}", ret_addr, self.next_addr, ret_addr);
					Some(ret_addr)
				},
			}
		}
		else
		{
			// No flow change
			self.next_addr += 2;
			Some(self.next_addr - 2)
		}
	}

	/// Issue a branch to trigger at the given location targeting the given
	/// address.
	pub fn branch(&mut self, location_addr: usize, target_addr: usize)
	{
		self.call_frame
			.branches
			.insert(location_addr, ControlFlowType::Branch(target_addr));
		self.report.issued_branches += 1;
	}

	/// Issue a branch to trigger at the given location targeting the given
	/// address.
	pub fn call(&mut self, location_addr: usize, target_addr: usize)
	{
		self.call_frame
			.branches
			.insert(location_addr, ControlFlowType::Call(target_addr));
		self.report.issued_calls += 1;
	}

	/// Issue a return to trigger at the given location
	pub fn ret(&mut self, location_addr: usize)
	{
		self.call_frame
			.branches
			.insert(location_addr, ControlFlowType::Return);
		self.report.issued_returns += 1;
	}
}
