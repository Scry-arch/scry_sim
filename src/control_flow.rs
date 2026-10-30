use crate::{
	data::{OperandStack, ProgramStack},
	metrics::MetricTracker,
	CallFrameState, ControlEvent, ControlFlowType, Event, EventTracker, ExecError, ExecState,
	IdEvent, Metric,
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
		// ID and type of branch including potential target
		(usize, ControlFlowType),
	>,
}
impl CallFrame
{
	pub fn set_state(&self, to_set: &mut CallFrameState)
	{
		to_set.ret_addr = self.ret_addr;
		to_set.branches = self
			.branches
			.iter()
			.map(|(loc, (_, ty))| (*loc, *ty))
			.collect();
	}

	pub fn from_state(state: &CallFrameState, events: &mut impl EventTracker) -> Self
	{
		Self {
			ret_addr: state.ret_addr,
			branches: state
				.branches
				.iter()
				.map(|(loc, ty)| (*loc, (events.new_control_id(), *ty)))
				.collect(),
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

	pub(crate) fn from_state(state: &ExecState, events: &mut impl EventTracker) -> Self
	{
		Self {
			next_addr: state.address,
			call_frame: CallFrame::from_state(&state.frame, events),
			call_stack: state.frame_stack.clone().into_iter().fold(
				VecDeque::new(),
				|mut stack, frame| {
					stack.push_back(CallFrame::from_state(&frame, events));
					stack
				},
			),
		}
	}

	/// Computes the address of the next instruction to execute
	///
	/// The given operand and program stacks are suitably updated if a call or
	/// return is triggered.
	///
	/// If the call stack is empty after a return is triggered, false is
	/// returned. Otherwise true
	pub fn next_addr(
		&mut self,
		op_stack: &mut OperandStack,
		prog_stack: &mut ProgramStack,
		tracker: &mut impl MetricTracker,
		event_tracker: &mut impl EventTracker,
	) -> Result<bool, ExecError>
	{
		if let Some((id, cft)) = self.call_frame.branches.remove(&self.next_addr)
		{
			use ControlFlowType::*;
			match cft
			{
				Branch(tar) =>
				{
					if let Some(earlier_trigger) =
						self.call_frame.branches.iter().find(|(k, _)| **k < tar)
					{
						return Err(ExecError::Exception(format!(
							"Branching past control flow trigger. Current: {:?}, trigger: {:?}, \
							 target: {:?}",
							self.next_addr, earlier_trigger.0, tar
						)));
					}
					self.next_addr = tar;
					tracker.add_stat(Metric::TriggeredBranches, 1);
					event_tracker.event(Event::Id(id, IdEvent::ControlTrigger(ControlEvent::Jump)))
				},
				Call(tar) =>
				{
					let old_next = self.next_addr;
					self.next_addr = tar;
					op_stack.push_queue();
					prog_stack.perform_call();
					let old_frame = std::mem::replace(
						&mut self.call_frame,
						CallFrame {
							ret_addr: old_next + 2,
							branches: HashMap::new(),
						},
					);
					self.call_stack.push_front(old_frame);
					tracker.add_stat(Metric::TriggeredCalls, 1);
					event_tracker.event(Event::Id(id, IdEvent::ControlTrigger(ControlEvent::Call)))
				},
				Return =>
				{
					let ret_addr = self.call_frame.ret_addr;
					self.next_addr = ret_addr;

					op_stack.pop_queue();
					if prog_stack.can_perform_return()
					{
						prog_stack.perform_return();
					}
					else
					{
						return Ok(false);
					}

					self.call_frame = if let Some(s) = self.call_stack.pop_front()
					{
						s
					}
					else
					{
						return Ok(false);
					};
					tracker.add_stat(Metric::TriggeredReturns, 1);
					event_tracker
						.event(Event::Id(id, IdEvent::ControlTrigger(ControlEvent::Return)))
				},
			}
		}
		else
		{
			// No flow change
			self.next_addr += 2;
		}
		Ok(true)
	}

	/// Issue a branch to trigger at the given location targeting the given
	/// address.
	pub fn branch(
		&mut self,
		location_addr: usize,
		target_addr: usize,
		tracker: &mut impl MetricTracker,
		event_tracker: &mut impl EventTracker,
	)
	{
		let id = event_tracker.new_control_id();
		self.call_frame
			.branches
			.insert(location_addr, (id, ControlFlowType::Branch(target_addr)));
		tracker.add_stat(Metric::IssuedBranches, 1);
		event_tracker.event(Event::Id(id, IdEvent::ControlIssue(ControlEvent::Jump)));
	}

	/// Issue a branch to trigger at the given location targeting the given
	/// address.
	pub fn call(
		&mut self,
		location_addr: usize,
		target_addr: usize,
		tracker: &mut impl MetricTracker,
		event_tracker: &mut impl EventTracker,
	)
	{
		let id = event_tracker.new_control_id();
		self.call_frame
			.branches
			.insert(location_addr, (id, ControlFlowType::Call(target_addr)));
		tracker.add_stat(Metric::IssuedCalls, 1);
		event_tracker.event(Event::Id(id, IdEvent::ControlIssue(ControlEvent::Call)));
	}

	/// Issue a return to trigger at the given location
	pub fn ret(
		&mut self,
		location_addr: usize,
		tracker: &mut impl MetricTracker,
		event_tracker: &mut impl EventTracker,
	)
	{
		let id = event_tracker.new_control_id();
		self.call_frame
			.branches
			.insert(location_addr, (id, ControlFlowType::Return));
		tracker.add_stat(Metric::IssuedReturns, 1);
		event_tracker.event(Event::Id(id, IdEvent::ControlIssue(ControlEvent::Return)))
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
