use crate::{
	memory::{MemError, Memory},
	value::Value,
	CallFrameState, ExecState, Metric, MetricTracker, OperandState, ValueType,
};
use num_traits::PrimInt;
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

	/// Constructs an operand that needs to read from the given address the
	/// given amount of scalars.
	///
	/// The type of the scalars is equivalent to the given type parameter.
	pub fn read<N: PrimInt>(address: usize, len: usize) -> Self
	{
		Self::read_typed(address, len, ValueType::new::<N>())
	}

	/// If this operand must still read its value from memory returns
	/// the address, number of scalars to read, and their type.
	///
	/// Otherwise return `None`.
	pub fn must_read(&self) -> Option<(usize, usize, ValueType)>
	{
		if let OperandState::MustRead((addr, len, typ)) = *(*self.op).borrow()
		{
			Some((addr, len, typ))
		}
		else
		{
			None
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
		match *(*self.op).borrow()
		{
			OperandState::MustRead((addr, len, typ)) =>
			{
				let new_value = f(addr, len, typ)?;
				*(*self.op).borrow_mut() = OperandState::Ready(new_value)
			},
			_ => (),
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

	/// Sets the operand's value.
	///
	/// Any operand connected to this one will also see the set value.
	pub fn set_value(&mut self, v: Value)
	{
		*(*self.op).borrow_mut() = OperandState::Ready(v);
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

/// The instruction input operand queue
#[derive(Debug)]
pub struct OperandQueue
{
	/// Stack of operand queues for (nested) callers of the current function
	stack: VecDeque<VecDeque<VecDeque<Operand>>>,

	/// Operand queue for the current function
	queue: VecDeque<VecDeque<Operand>>,

	/// The operands that should be used by the next instruction
	ready: VecDeque<Operand>,
}
impl OperandQueue
{
	pub fn new(ready_ops: impl Iterator<Item = Value>) -> Self
	{
		Self {
			stack: VecDeque::new(),
			queue: VecDeque::new(),
			ready: VecDeque::from_iter(ready_ops.map(|v| v.into())),
		}
	}

	/// An iterator providing the operands at the front of the operand queue.
	/// The given memory is used to perform any pending read operations.
	///
	/// Only when `next` is called, will a returned operand be treated as
	/// "consumed" by the report.
	///
	/// When the iterator is dropped, the queue discards the remaining operands
	/// at the front of the queue, and moves the next set of operands to the
	/// front.
	pub fn ready_iter<'a, M: Memory, T: MetricTracker>(
		&'a mut self,
		mem: &'a mut M,
		tracker: &'a mut T,
	) -> impl 'a + Iterator<Item = (Value, Option<(MemError, usize)>)>
	{
		struct RemoveIter<'b, M: Memory, Ti: MetricTracker>
		{
			queue: &'b mut OperandQueue,
			mem: &'b mut M,
			tracker: &'b mut Ti,
		}
		impl<'b, M: Memory, Ti: MetricTracker> Iterator for RemoveIter<'b, M, Ti>
		{
			type Item = (Value, Option<(MemError, usize)>);

			fn next(&mut self) -> Option<Self::Item>
			{
				self.queue.ready.pop_front().and_then(|mut op| {
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
				self.queue.ready = self.queue.queue.pop_front().unwrap_or(VecDeque::new());
			}
		}
		RemoveIter {
			queue: self,
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

	/// Pushes the given operand to the back of the queue at the given index.
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

	/// Moves the operand found on the queue with index `src_q_idx`, position
	/// `src_idx` in that queue, to the back of the queue with idx 'target_idx'
	pub fn reorder(
		&mut self,
		src_q_idx: usize,
		src_idx: usize,
		target_idx: usize,
		tracker: &mut impl MetricTracker,
	)
	{
		let op = self
			.queue
			.get_mut(src_q_idx)
			.and_then(|q| q.remove(src_idx))
			.map(|op| {
				tracker.add_stat(Metric::ReorderedOperands, 1);
				op
			})
			.unwrap_or(Value::new_nar::<u8>(0).into());
		self.push_op_unreported(target_idx, op);
	}

	/// Moves all operands on the ready queue to the back of the `dest_idx`
	/// queue.
	pub fn reorder_ready(&mut self, dest_idx: usize, tracker: &mut impl MetricTracker)
	{
		while let Some(op) = self.ready.remove(0).map(|op| {
			tracker.add_stat(Metric::ReorderedOperands, 1);
			op
		})
		{
			self.push_op_unreported(dest_idx, op);
		}
	}

	/// Pushes the current operand queue onto the queue stack keeping the ready
	/// queue in place.
	pub fn push_queue(&mut self)
	{
		self.stack
			.push_back(std::mem::replace(&mut self.queue, VecDeque::new()));
	}

	/// Pops the top operand queue from the queue stack keeping the ready queue
	/// in place. The popped queue becomes the current queue, discarding the old
	/// queue.
	///
	/// If the stack is empty, the current queue becomes empty.
	pub fn pop_queue(&mut self)
	{
		self.queue = self.stack.pop_front().unwrap_or(VecDeque::new());

		// Pop front of queue into front of ready-queue.
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
		let mut op_queues = HashMap::new();

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
			(all_ops.remove(0), all_ops)
		};
		from.enumerate().for_each(|(idx, q)| {
			if q.len() > 0
			{
				op_queues.insert(idx, convert_queue(q));
			}
		});

		to_set.op_queues = op_queues;
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
impl<'a> From<&'a ExecState> for OperandQueue
{
	fn from(state: &'a ExecState) -> Self
	{
		let frame_op_queues = |frame: &CallFrameState| {
			let reads: Vec<_> = frame
				.reads
				.clone()
				.into_iter()
				.map(|(addr, len, typ)| Operand::read_typed(addr, len, typ))
				.collect();
			frame
				.op_queues
				.iter()
				.fold(VecDeque::new(), |mut queues, (q_idx, (f, q))| {
					let ops = once(f).chain(q.iter()).cloned().map(|op| {
						match op
						{
							OperandState::MustRead(idx) => reads[idx].clone(),
							OperandState::Ready(v) => v.clone().into(),
						}
					});
					if queues.len() <= *q_idx
					{
						queues.resize_with(*q_idx + 1, VecDeque::new);
					}
					queues.get_mut(*q_idx).unwrap().extend(ops);
					queues
				})
		};
		let mut curr_frame_op_queues = frame_op_queues(&state.frame);
		let first_queue = curr_frame_op_queues.pop_front().unwrap_or(VecDeque::new());
		Self {
			stack: state.frame_stack.clone().into_iter().fold(
				VecDeque::new(),
				|mut stack, frame| {
					stack.push_back(frame_op_queues(&frame));
					stack
				},
			),
			queue: curr_frame_op_queues,
			ready: first_queue,
		}
	}
}
