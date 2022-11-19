use crate::{
	memory::{MemError, Memory},
	value::Value,
	CallFrameState, ExecState, Metric, MetricTracker, OperandList, OperandState, ValueType,
};
use std::{
	cell::{Ref, RefCell},
	collections::{HashMap, VecDeque},
	fmt::Debug,
	iter::{once, FromIterator},
	ops::Deref,
	rc::Rc,
};

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
	op: Rc<RefCell<OperandState<(usize, usize, ValueType)>>>,
}
impl Operand
{
	/// Constructs an operand that needs to read from the given address the
	/// given amount of scalars of the given type.
	pub fn read_typed(address: usize, len: usize, typ: ValueType) -> Self
	{
		Self {
			op: Rc::new(OperandState::MustRead((address, len, typ)).into()),
		}
	}

	/// Returns a reference to this operands value if available.
	///
	/// Otherwise, calls the given function with the address, number of scalars,
	/// and type to read from memory. The function performs the read and returns
	/// the resulting value, which is saved in the operand and a reference to it
	/// returned. If the function fails, the error is returned.
	pub(crate) fn get_value_or<E, F: FnOnce(usize, usize, ValueType) -> Result<Value, E>>(
		&mut self,
		f: F,
	) -> Result<impl '_ + Deref<Target = Value>, E>
	{
		if self.get_value().is_none()
		{
			let mut ref_mut = (*self.op).borrow_mut();
			match *ref_mut
			{
				OperandState::MustRead((addr, len, typ)) =>
				{
					let new_value = f(addr, len, typ)?;
					*ref_mut = OperandState::Ready(new_value)
				},
				_ => (),
			}
		}

		if let Some(v) = self.get_value()
		{
			Ok(v)
		}
		else
		{
			unreachable!()
		}
	}

	/// Returns a reference to this operands value if available.
	pub fn get_value(&self) -> Option<impl '_ + Deref<Target = Value>>
	{
		if let OperandState::Ready(_) = *(*self.op).borrow()
		{
			Some(Ref::map((*self.op).borrow(), |b| {
				match b
				{
					OperandState::Ready(v) => v,
					_ => unreachable!(),
				}
			}))
		}
		else
		{
			None
		}
	}
}
impl From<Value> for Operand
{
	fn from(v: Value) -> Self
	{
		Self {
			op: Rc::new(OperandState::Ready(v).into()),
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
}
impl OperandStack
{
	pub fn new(ready_ops: impl Iterator<Item = Value>) -> Self
	{
		Self {
			stack: VecDeque::new(),
			queue: VecDeque::new(),
			ready: VecDeque::from_iter(ready_ops.map(|v| v.into())),
		}
	}

	/// An iterator providing the operand list at the front of the operand
	/// queue. The given memory is used to perform any pending read operations.
	///
	/// Only when `next` is called, will a returned operand be treated as
	/// "consumed" by the report.
	///
	/// When the iterator is dropped, the queue discards the remaining operands
	/// at the front of the queue, and moves the next operand list to the front.
	pub fn ready_iter<'a, M: Memory, T: MetricTracker>(
		&'a mut self,
		mem: &'a mut M,
		tracker: &'a mut T,
	) -> impl 'a + Iterator<Item = (Value, Option<(MemError, usize)>)>
	{
		struct RemoveIter<'b, M: Memory, Ti: MetricTracker>
		{
			stack: &'b mut OperandStack,
			mem: &'b mut M,
			tracker: &'b mut Ti,
		}
		impl<'b, M: Memory, Ti: MetricTracker> Iterator for RemoveIter<'b, M, Ti>
		{
			type Item = (Value, Option<(MemError, usize)>);

			fn next(&mut self) -> Option<Self::Item>
			{
				self.stack.ready.pop_front().and_then(|mut op: Operand| {
					let mut err = None;
					let val = op
						.get_value_or::<(), _>(|addr, len, typ| {
							let mut v = Value::new_nar_typed(typ, 0);
							err = self.mem.read_data(addr, &mut v, len, self.tracker).err();
							Ok(v)
						})
						.unwrap()
						.clone();
					self.tracker.add_stat(Metric::ConsumedOperands, 1);
					self.tracker.add_stat(Metric::ConsumedBytes, val.size());
					Some((val, err))
				})
			}
		}
		impl<'b, M: Memory, Ti: MetricTracker> Drop for RemoveIter<'b, M, Ti>
		{
			fn drop(&mut self)
			{
				self.stack.ready = self.stack.queue.pop_front().unwrap_or(VecDeque::new());
			}
		}
		RemoveIter {
			stack: self,
			mem,
			tracker,
		}
	}

	/// An iterator peeking at the operands at the front of the operand queue.
	pub fn ready_peek(&self) -> impl Iterator<Item = &Operand>
	{
		self.ready.iter()
	}

	/// Pushes the given operand to the back of the queue at the given index.
	///
	/// Doesn't update any report data
	fn push_op_unreported(&mut self, mut idx: usize, op: Operand)
	{
		if idx == 0
		{
			self.ready.push_back(op);
		}
		else
		{
			idx -= 1;
			// Ensures the operand queue is long enough to include the given index.
			if self.queue.len() <= idx
			{
				self.queue.resize_with(idx + 1, || VecDeque::new());
			}
			self.queue[idx].push_back(op);
		}
	}

	/// Pushes the given operand to the back of the list at the given index.
	pub fn push_operand(&mut self, idx: usize, op: Operand, tracker: &mut impl MetricTracker)
	{
		if let Some(v) = op.get_value()
		{
			tracker.add_stat(Metric::QueuedValues, 1);
			tracker.add_stat(Metric::QueuedValueBytes, v.size());
		}
		else
		{
			tracker.add_stat(Metric::QueuedReads, 1);
		}
		self.push_op_unreported(idx, op);
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
		tracker: &mut impl MetricTracker,
	)
	{
		once(&mut self.ready)
			.chain(self.queue.iter_mut())
			.nth(src_list_idx)
			.and_then(|list| list.remove(src_idx))
			.map(|op| {
				tracker.add_stat(Metric::ReorderedOperands, 1);
				self.push_op_unreported(target_idx, op);
			});
	}

	/// Moves all operands on the list with at index `src_idx` to the back of
	/// the `dest_idx` queue.
	pub fn reorder_list(
		&mut self,
		src_idx: usize,
		dest_idx: usize,
		tracker: &mut impl MetricTracker,
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
			tracker.add_stat(Metric::ReorderedOperands, 1);
			op
		})
		{
			self.push_op_unreported(dest_idx, op);
		}
	}

	/// Moves all operands on the ready list to the back of the `dest_idx`
	/// queue.
	pub fn reorder_ready(&mut self, dest_idx: usize, tracker: &mut impl MetricTracker)
	{
		self.reorder_list(0, dest_idx, tracker)
	}

	/// Pushes the current operand queue onto the queue stack keeping the ready
	/// list in place.
	pub fn push_queue(&mut self)
	{
		self.stack
			.push_back(std::mem::replace(&mut self.queue, VecDeque::new()));
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
		let mut reads = Vec::new();
		let mut op_queue = HashMap::new();

		let mut convert_queue = |q: &VecDeque<Operand>| {
			let mut all_ops = q
				.iter()
				.map(|op| {
					match (*op.op).borrow().deref()
					{
						OperandState::MustRead(_) =>
						{
							OperandState::MustRead(
								if let Some((idx, _)) = reads
									.iter()
									.enumerate()
									.find(|(_, read_op)| Rc::ptr_eq(read_op, &op.op))
								{
									idx
								}
								else
								{
									reads.push(op.op.clone());
									reads.len() - 1
								},
							)
						},
						OperandState::Ready(v) => OperandState::Ready(v.clone()),
					}
				})
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
		to_set.reads = reads
			.iter()
			.map(|op| {
				if let OperandState::MustRead(addr_len_typ) = (*op).borrow().deref()
				{
					*addr_len_typ
				}
				else
				{
					unreachable!()
				}
			})
			.collect();
	}

	pub fn set_all_frame_states<'a>(&self, to_set: impl Iterator<Item = &'a mut CallFrameState>)
	{
		to_set
			.enumerate()
			.for_each(|(idx, frame)| self.set_frame_state(idx, frame));
	}
}
/// Constructs an OperandQueue equivalent to an execution state
impl<'a> From<&'a ExecState> for OperandStack
{
	fn from(state: &'a ExecState) -> Self
	{
		let frame_op_queue = |frame: &CallFrameState| {
			let reads: Vec<_> = frame
				.reads
				.clone()
				.into_iter()
				.map(|(addr, len, typ)| Operand::read_typed(addr, len, typ))
				.collect();
			frame
				.op_queue
				.iter()
				.fold(VecDeque::new(), |mut op_lists, (list_idx, op_list)| {
					let ops = op_list.iter().cloned().map(|op| {
						match op
						{
							OperandState::MustRead(idx) => reads[idx].clone(),
							OperandState::Ready(v) => v.clone().into(),
						}
					});
					if op_lists.len() <= *list_idx
					{
						op_lists.resize_with(*list_idx + 1, VecDeque::new);
					}
					op_lists.get_mut(*list_idx).unwrap().extend(ops);
					op_lists
				})
		};
		let mut curr_frame_op_queue = frame_op_queue(&state.frame);
		let first_list = curr_frame_op_queue.pop_front().unwrap_or(VecDeque::new());
		Self {
			stack: state.frame_stack.clone().into_iter().fold(
				VecDeque::new(),
				|mut stack, frame| {
					stack.push_back(frame_op_queue(&frame));
					stack
				},
			),
			queue: curr_frame_op_queue,
			ready: first_list,
		}
	}
}
