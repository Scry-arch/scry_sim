use crate::{
	value::Value, CallFrameState, Event, EventTracker, ExecState, IdEvent, OperandEvent,
	OperandList, StackFrame,
};
use std::{
	cell::RefCell,
	collections::{HashMap, VecDeque},
	fmt::Debug,
	iter::{once, FromIterator},
	mem::replace,
	ops::Deref,
	rc::Rc,
};

/// The highest number of inputs a non-call/dataflow instruction takes.
///
/// This is currently set to the maximum inputs a
/// [`scry_isa::Instruction::Store`] takes:
/// 1. value to store
/// 2. base address
/// 3. Index/offset amount
pub const MAX_INSTR_INPUTS: usize = 3;

/// An instruction operand.
///
/// Differs from values in that they may not actually be available yet or may be
/// connected to other operands that use the same value.
/// This can happen during a read from memory. Initially the read is issued but
/// not yet needed. The operand may then be cloned e.g. by a duplicate
/// instruction. The resulting operands are then connected, such that when one
/// of them is consumed by an instruction, e.g. an add, the memory
/// read is actually performed and all connected operands now used the read
/// value.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Operand
{
	pub id: usize,
	op: Rc<RefCell<Value>>,
}
impl Operand
{
	/// Returns a reference to this operands value if available.
	pub fn get_value(&self) -> impl '_ + Deref<Target = Value>
	{
		(*self.op).borrow()
	}

	pub fn from_value(v: Value, id: usize) -> Self
	{
		Self {
			id,
			op: Rc::new(v.into()),
		}
	}
}

/// The instruction input operand stack
#[derive(Debug)]
pub struct OperandStack
{
	/// Stack of operand queues for (nested) callers of the current function
	stack: VecDeque<VecDeque<VecDeque<Operand>>>,

	/// Operand queue for the current function
	queue: VecDeque<VecDeque<Operand>>,

	/// The operands that should be used by the next instruction
	ready: VecDeque<Operand>,

	/// The additional depth beyond [`MAX_INSTR_INPUTS`]
	more_depth: usize,
}
impl OperandStack
{
	pub fn new(ready_ops: impl Iterator<Item = Operand>, more_depth: usize) -> Self
	{
		Self {
			stack: VecDeque::new(),
			queue: VecDeque::new(),
			ready: VecDeque::from_iter(ready_ops),
			more_depth,
		}
	}

	pub fn op_list_depth(&self) -> usize
	{
		MAX_INSTR_INPUTS + self.more_depth
	}

	/// An iterator providing the operand list at the front of the operand
	/// queue.
	///
	/// Only when `next` is called, will a returned operand be reported as
	/// consumed to the event tracker.
	///
	/// When the iterator is dropped, the queue discards the remaining operands
	/// at the front of the queue, and moves the next operand list to the front.
	pub fn ready_iter<'a, E: EventTracker>(
		&'a mut self,
		events: &'a mut E,
	) -> impl 'a + Iterator<Item = Value>
	{
		struct RemoveIter<'b, Ei: EventTracker>
		{
			stack: &'b mut OperandStack,
			events: &'b mut Ei,
		}
		impl<'b, Ei: EventTracker> Iterator for RemoveIter<'b, Ei>
		{
			type Item = Value;

			fn next(&mut self) -> Option<Self::Item>
			{
				self.stack.pop_ready(self.events)
			}
		}
		impl<'b, Ei: EventTracker> Drop for RemoveIter<'b, Ei>
		{
			fn drop(&mut self)
			{
				self.stack.ready.iter().for_each(|op| {
					self.events
						.event(Event::Id(op.id, IdEvent::Operand(OperandEvent::Discard)))
				});
				self.stack.ready = self.stack.queue.pop_front().unwrap_or(VecDeque::new());
			}
		}
		RemoveIter {
			stack: self,
			events,
		}
	}

	/// Extracts the first operand on the ready queue
	pub fn pop_ready<E: EventTracker>(&mut self, event_tracker: &mut E) -> Option<Value>
	{
		self.ready.pop_front().and_then(|op: Operand| {
			let val = op.get_value().clone();
			event_tracker.event(Event::Id(op.id, IdEvent::Operand(OperandEvent::Consume)));
			Some(val)
		})
	}

	/// An iterator peeking at the operands at the front of the operand queue.
	pub fn ready_peek(&self) -> impl Iterator<Item = &Operand>
	{
		self.ready.iter()
	}

	/// Pushes the given operand to the back of the queue at the given index if
	/// the queue has not reached its operand limit. Returns whether the operand
	/// was pushed. I.e. returns false if the queue is at its limit
	///
	/// Doesn't report any events
	fn push_op_unreported(&mut self, mut idx: usize, op: Operand) -> bool
	{
		if idx == 0
		{
			if self.ready.len() < self.op_list_depth()
			{
				self.ready.push_back(op);
				true
			}
			else
			{
				false
			}
		}
		else
		{
			idx -= 1;
			// Ensures the operand queue is long enough to include the given
			// index.
			if self.queue.len() <= idx
			{
				self.queue.resize_with(idx + 1, || VecDeque::new());
			}
			if self.queue[idx].len() < self.op_list_depth()
			{
				self.queue[idx].push_back(op);
				true
			}
			else
			{
				false
			}
		}
	}

	/// Pushes the given operand to the back of the list at the given index.
	pub fn push_operand(&mut self, idx: usize, op: Operand, event_tracker: &mut impl EventTracker)
	{
		// We report the operand as produced regardless of whether it is
		// actually queued. If it is put on a full operand list and therefore
		// dropped, a discard event is additionally reported.
		let id = op.id;
		event_tracker.event(Event::Id(
			id,
			IdEvent::Operand(OperandEvent::Produce(op.get_value().typ.into())),
		));
		if !self.push_op_unreported(idx, op)
		{
			event_tracker.event(Event::Id(id, IdEvent::Operand(OperandEvent::Discard)));
		}
	}

	/// Moves the operand found on the operand list with index `src_list_idx`,
	/// position `src_idx` in that list, to the back of the list with idx
	/// 'target_idx'
	///
	/// If no operand is present at the location, nothing happens
	pub fn reorder(
		&mut self,
		src_list_idx: usize,
		src_idx: usize,
		target_idx: usize,
		event_tracker: &mut impl EventTracker,
	)
	{
		once(&mut self.ready)
			.chain(self.queue.iter_mut())
			.nth(src_list_idx)
			.and_then(|list| list.remove(src_idx))
			.map(|op| {
				let id = op.id;
				event_tracker.event(Event::Id(id, IdEvent::Operand(OperandEvent::Forward)));
				if !self.push_op_unreported(target_idx, op)
				{
					event_tracker.event(Event::Id(id, IdEvent::Operand(OperandEvent::Discard)));
				}
			});
	}

	/// Moves all operands on the list with at index `src_idx` to the back of
	/// the `dest_idx` queue.
	pub fn reorder_list(
		&mut self,
		src_idx: usize,
		dest_idx: usize,
		ev_tracker: &mut impl EventTracker,
	)
	{
		if src_idx == dest_idx
		{
			return;
		}

		while let Some(op) = if src_idx == 0
		{
			Some(&mut self.ready)
		}
		else
		{
			self.queue.get_mut(src_idx - 1)
		}
		.and_then(|list| list.remove(0))
		.map(|op| {
			ev_tracker.event(Event::Id(op.id, IdEvent::Operand(OperandEvent::Forward)));
			op
		})
		{
			let id = op.id;
			if !self.push_op_unreported(dest_idx, op)
			{
				ev_tracker.event(Event::Id(id, IdEvent::Operand(OperandEvent::Discard)));
			}
		}
	}

	/// Moves all operands on the ready list to the back of the `dest_idx`
	/// queue.
	pub fn reorder_ready(&mut self, dest_idx: usize, events: &mut impl EventTracker)
	{
		self.reorder_list(0, dest_idx, events)
	}

	/// Pushes the current operand queue onto the queue stack keeping the ready
	/// list in place.
	pub fn push_queue(&mut self)
	{
		self.stack
			.push_front(std::mem::replace(&mut self.queue, VecDeque::new()));
	}

	/// Pops the top operand queue from the queue stack keeping the ready list
	/// in place. The popped queue becomes the current queue, discarding the old
	/// queue.
	///
	/// If the stack is empty, the current queue becomes empty.
	pub fn pop_queue(&mut self)
	{
		self.queue = self.stack.pop_front().unwrap_or(VecDeque::new());

		// Pop front of queue into front of ready list.
		// This ensures operands sent to the instruction after a call
		// are at the front of the queue after the return
		if let Some(ops) = self.queue.pop_front()
		{
			ops.into_iter()
				.rev()
				.for_each(|op| self.ready.push_front(op));
			self.ready.truncate(self.op_list_depth());
		}
	}

	pub fn set_frame_state(&self, idx: usize, to_set: &mut CallFrameState)
	{
		let from = if idx == 0
		{
			Some(&self.ready).into_iter().chain(self.queue.iter())
		}
		else
		{
			None.into_iter().chain(self.stack.get(idx - 1).unwrap())
		};
		let mut op_queue = HashMap::new();

		let convert_queue = |q: &VecDeque<Operand>| {
			let mut all_ops = q
				.iter()
				.map(|op| op.get_value().clone())
				.collect::<Vec<_>>();
			OperandList::new(all_ops.remove(0), all_ops)
		};
		from.enumerate().for_each(|(idx, q)| {
			if q.len() > 0
			{
				op_queue.insert(idx, convert_queue(q));
			}
		});

		to_set.op_queue = op_queue;
	}

	pub fn set_all_frame_states<'a>(&self, to_set: impl Iterator<Item = &'a mut CallFrameState>)
	{
		to_set
			.enumerate()
			.for_each(|(idx, frame)| self.set_frame_state(idx, frame));
	}

	/// Calculates the effective address of a value with given size and index in
	/// the stack with the given stack base. I.e. returns an absolute address
	pub fn operand_stack_address(stack_base: usize, size: usize, index: usize) -> usize
	{
		let stack_offset = index * size;
		let missing_align = size - (stack_base % size);
		stack_offset
			+ if missing_align != size
			{
				stack_base + missing_align
			}
			else
			{
				stack_base
			}
	}

	/// Constructs an OperandQueue equivalent to an execution state and using
	/// the given tracker for assigning IDs
	pub(crate) fn from_exec_state<E: EventTracker>(state: &ExecState, tracker: &mut E) -> Self
	{
		let frame_op_queue = |frame: &CallFrameState, tracker: &mut E| {
			frame
				.op_queue
				.iter()
				.fold(VecDeque::new(), |mut op_lists, (list_idx, op_list)| {
					let ops = op_list
						.iter()
						.cloned()
						.map(|op| Operand::from_value(op, tracker.new_id()));
					if op_lists.len() <= *list_idx
					{
						op_lists.resize_with(*list_idx + 1, VecDeque::new);
					}
					op_lists.get_mut(*list_idx).unwrap().extend(ops);
					op_lists
				})
		};
		let mut curr_frame_op_queue = frame_op_queue(&state.frame, tracker);
		let first_list = curr_frame_op_queue.pop_front().unwrap_or(VecDeque::new());
		Self {
			stack: state.frame_stack.clone().into_iter().fold(
				VecDeque::new(),
				|mut stack, frame| {
					stack.push_back(frame_op_queue(&frame, tracker));
					stack
				},
			),
			queue: curr_frame_op_queue,
			ready: first_list,
			more_depth: state.more_depth,
		}
	}
}

/// The program stack
#[derive(Debug)]
pub struct ProgramStack
{
	top: StackFrame,
	rest: VecDeque<StackFrame>,
}
impl ProgramStack
{
	pub fn new(top: StackFrame) -> Self
	{
		Self {
			top,
			rest: VecDeque::new(),
		}
	}

	pub fn top(&self) -> &StackFrame
	{
		&self.top
	}

	pub fn top_mut(&mut self) -> &mut StackFrame
	{
		&mut self.top
	}

	pub fn perform_call(&mut self)
	{
		let new_top = self.top.frame_call();
		self.rest.push_front(replace(&mut self.top, new_top));
	}

	pub fn can_perform_return(&self) -> bool
	{
		!self.rest.is_empty()
	}

	pub fn perform_return(&mut self)
	{
		assert!(self.can_perform_return());
		let old_top = replace(&mut self.top, self.rest.pop_front().unwrap());
		self.top.frame_return(old_top);
	}

	pub fn set_frame_state(&self, idx: usize, to_set: &mut CallFrameState)
	{
		to_set.stack = if idx == 0
		{
			self.top.clone()
		}
		else
		{
			self.rest[idx - 1].clone()
		};
	}

	pub fn set_all_frame_states<'a>(&self, to_set: impl Iterator<Item = &'a mut CallFrameState>)
	{
		to_set
			.enumerate()
			.for_each(|(idx, frame)| self.set_frame_state(idx, frame));
	}
}
/// Constructs a ProgramStack equivalent to an execution state
impl<'a> From<&'a ExecState> for ProgramStack
{
	fn from(state: &'a ExecState) -> Self
	{
		let top = state.frame.stack.clone();
		let rest = state
			.frame_stack
			.iter()
			.map(|c| &c.stack)
			.cloned()
			.collect();
		Self { top, rest }
	}
}
