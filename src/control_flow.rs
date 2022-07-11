use crate::{
	data::OperandQueue, metrics::MetricTracker, CallFrameState, ControlFlowType, ExecState, Metric,
};
use std::collections::{HashMap, VecDeque};

#[derive(Debug)]
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
impl CallFrame
{
	pub fn set_state(&self, to_set: &mut CallFrameState)
	{
		to_set.ret_addr = self.ret_addr;
		to_set.branches = self.branches.clone();
	}
}
impl<'a> From<&'a CallFrameState> for CallFrame
{
	fn from(state: &'a CallFrameState) -> Self
	{
		Self {
			ret_addr: state.ret_addr,
			branches: state.branches.clone(),
		}
	}
}

/// Controls the execution of the program.
#[derive(Debug)]
pub struct ControlFlow
{
	/// Next instruction to be executed
	pub next_addr: usize,
	call_frame: CallFrame,
	call_stack: VecDeque<CallFrame>,
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
		}
	}

	/// Computes the address of the next instruction to execute
	///
	/// The given queue is suitably updated if a call or return is triggered.
	///
	/// If the call stack is empty after a return is triggered, false is
	/// returned. Otherwise true
	pub fn next_addr(&mut self, queue: &mut OperandQueue, tracker: &mut impl MetricTracker)
		-> bool
	{
		if let Some(cft) = self.call_frame.branches.remove(&self.next_addr)
		{
			use ControlFlowType::*;
			match cft
			{
				Branch(tar) =>
				{
					self.next_addr = tar;
					tracker.add_stat(Metric::TriggeredBranches, 1);
				},
				Call(tar) =>
				{
					let old_next = self.next_addr;
					self.next_addr = tar;
					queue.push_queue();
					let old_frame = std::mem::replace(
						&mut self.call_frame,
						CallFrame {
							ret_addr: old_next + 2,
							branches: HashMap::new(),
						},
					);
					self.call_stack.push_back(old_frame);
					tracker.add_stat(Metric::TriggeredCalls, 1);
				},
				Return =>
				{
					let ret_addr = self.call_frame.ret_addr;
					self.next_addr = ret_addr;

					queue.pop_queue(tracker);
					self.call_frame = if let Some(s) = self.call_stack.pop_back()
					{
						s
					}
					else
					{
						return false;
					};
					tracker.add_stat(Metric::TriggeredReturns, 1);
				},
			}
		}
		else
		{
			// No flow change
			self.next_addr += 2;
		}
		true
	}

	/// Issue a branch to trigger at the given location targeting the given
	/// address.
	pub fn branch(
		&mut self,
		location_addr: usize,
		target_addr: usize,
		tracker: &mut impl MetricTracker,
	)
	{
		self.call_frame
			.branches
			.insert(location_addr, ControlFlowType::Branch(target_addr));
		tracker.add_stat(Metric::IssuedBranches, 1);
	}

	/// Issue a branch to trigger at the given location targeting the given
	/// address.
	pub fn call(
		&mut self,
		location_addr: usize,
		target_addr: usize,
		tracker: &mut impl MetricTracker,
	)
	{
		self.call_frame
			.branches
			.insert(location_addr, ControlFlowType::Call(target_addr));
		tracker.add_stat(Metric::IssuedCalls, 1);
	}

	/// Issue a return to trigger at the given location
	pub fn ret(&mut self, location_addr: usize, tracker: &mut impl MetricTracker)
	{
		self.call_frame
			.branches
			.insert(location_addr, ControlFlowType::Return);
		tracker.add_stat(Metric::IssuedReturns, 1);
	}

	/// Sets the given call frame states' return address and pending control
	/// flow to the equivalent of the current control flow.
	///
	/// The given list is assumed to be in order, where the 0'th element is the
	/// current call frame.
	pub fn set_frame_state(&self, to_set: &mut Vec<CallFrameState>)
	{
		let set_idx = |vec: &mut Vec<_>, idx, frame: &CallFrame| {
			if vec.len() == idx
			{
				vec.push(Default::default());
			}
			assert!(vec.len() > idx);
			if let Some(state) = vec.get_mut(idx)
			{
				frame.set_state(state);
			}
			else
			{
				unreachable!()
			}
		};
		set_idx(to_set, 0, &self.call_frame);
		self.call_stack
			.iter()
			.enumerate()
			.for_each(|(i, f)| set_idx(to_set, i + 1, f));
	}
}
/// Constructions a ControlFlow equivalent to an execution state
impl<'a> From<&'a ExecState> for ControlFlow
{
	fn from(state: &'a ExecState) -> Self
	{
		Self {
			next_addr: state.address,
			call_frame: (&state.frame).into(),
			call_stack: state.frame_stack.clone().into_iter().fold(
				VecDeque::new(),
				|mut stack, frame| {
					stack.push_back((&frame).into());
					stack
				},
			),
		}
	}
}
