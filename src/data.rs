use crate::memory::{MemError, Memory};
use num_traits::PrimInt;
use std::{
	cell::{RefCell, RefMut},
	collections::VecDeque,
	iter::FromIterator,
	rc::Rc,
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ValueType
{
	/// Unsigned integer with power of 2 bytes
	Uint(u8),

	/// Signed integer with power of 2 bytes
	Int(u8),
}

impl ValueType
{
	fn new_typed<N: PrimInt>() -> Self
	{
		let size = std::mem::size_of::<N>();
		assert!(size <= u8::MAX as usize);
		assert_eq!(size.count_ones(), 1);
		let pow2_size = size.trailing_zeros();
		if N::min_value() < N::zero()
		{
			ValueType::Int(pow2_size as u8)
		}
		else
		{
			ValueType::Uint(pow2_size as u8)
		}
	}
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ValueState
{
	/// Value with given data
	Val(Box<[u8]>),

	/// Not-A-Result: Signifies something went wrong.
	///
	/// Given is the address of the instruction that caused it.
	Nar(usize),

	/// Not-A-Number: Signifies the intentional lack of a value.
	Nan,
}

impl ValueState
{
	pub fn set_val(&mut self, bytes: &[u8])
	{
		if let ValueState::Val(val) = self
		{
			assert_eq!(val.len(), bytes.len());
			val.iter_mut().zip(bytes).for_each(|(v, b)| *v = *b);
		}
		else
		{
			*self = ValueState::Val(Vec::from(bytes).into_boxed_slice());
		}
	}
}

/// A piece of data.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Value
{
	/// Scalar type of the value
	typ: ValueType,

	/// Vector length
	data: Box<[ValueState]>,
}

impl Value
{
	pub fn new(typ: ValueType) -> Self
	{
		Self {
			typ,
			data: Vec::from([]).into_boxed_slice(),
		}
	}

	pub fn new_nan<N: PrimInt>() -> Self
	{
		let mut result: Value = N::zero().into();
		result.data = Vec::from([ValueState::Nan]).into_boxed_slice();
		result
	}

	pub fn new_nar<N: PrimInt>(payload: usize) -> Self
	{
		let mut result: Value = N::zero().into();
		result.data = Vec::from([ValueState::Nar(payload)]).into_boxed_slice();
		result
	}

	pub fn new_nan_typed(typ: ValueType) -> Self
	{
		let mut result = Value::new(typ);
		result.data = Vec::from([ValueState::Nan]).into_boxed_slice();
		result
	}

	pub fn new_nar_typed(typ: ValueType, payload: usize) -> Self
	{
		let mut result = Value::new(typ);
		result.data = Vec::from([ValueState::Nar(payload)]).into_boxed_slice();
		result
	}

	pub fn value_type(&self) -> ValueType
	{
		self.typ
	}

	/// Returns the number of bytes one scalar element if this value's type
	/// takes up
	pub fn scale(&self) -> usize
	{
		match self.typ
		{
			ValueType::Uint(x) | ValueType::Int(x) => 2usize.pow(x as u32),
		}
	}

	/// Returns the vector length of this value
	pub fn len(&self) -> usize
	{
		// Ensure array is multiple of data type size
		assert_eq!(self.data.len() % self.scale(), 0);
		self.data.len() / self.scale()
	}

	/// Returns the number of bytes this value takes up
	pub fn size(&self) -> usize
	{
		self.data.len()
	}

	/// Returns the states of each element in the value
	pub fn iter(&self) -> impl Iterator<Item = &ValueState>
	{
		self.data.iter()
	}

	/// Returns the mutable states of each element in the value.
	///
	/// The caller must ensure only valid changes are made.
	pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut ValueState>
	{
		self.data.iter_mut()
	}
}

impl<N: PrimInt> From<N> for Value
{
	fn from(num: N) -> Self
	{
		let le = num.to_le();
		let bytes = unsafe {
			std::slice::from_raw_parts(
				std::mem::transmute::<&N, *const u8>(&le),
				std::mem::size_of::<N>(),
			)
		};

		let mut result = Self::new(ValueType::new_typed::<N>());
		result.data = Vec::from([ValueState::Val(Box::from(bytes))]).into_boxed_slice();
		result
	}
}

/// An instruction operand
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Operand
{
	/// A value ready to be used in instructions as is.
	Val(Value),

	/// A value that needs to be read from memory before first use.
	///
	/// If ok, value has been read and is ready for use.
	/// If Err, the first element is the address to read from,
	/// and the second is the number of scalars to read
	Read(Rc<RefCell<Result<Value, (usize, usize, ValueType)>>>),
}

impl Operand
{
	fn is_non_ready_read(&self) -> bool
	{
		if let Operand::Read(op) = self
		{
			if let Err(_) = *(op.borrow())
			{
				return true;
			}
		}
		false
	}
}

#[derive(Debug)]
struct ReportData
{
	/// How many operands at the head of the queue were consumed.
	ready_consumed: usize,

	/// The number of bytes consumed at the head of the queue
	ready_consumed_bytes: usize,

	/// Number of operands added to queue anywhere
	added: usize,

	/// Amount of bytes added to the queue anywhere.
	///
	/// This counts only complete values added and not any read operands.
	added_bytes: usize,

	/// Amount of bytes read by read operands.
	///
	/// Read operands that are duplicated only count once.
	/// Any read operands that are discarded before their first consumption
	/// don't increase this count (the duplicates instead count in
	/// 'ready_consumed_bytes').
	added_read_bytes: usize,

	/// How many pending reads that were discarded
	discarded_non_ready_reads: usize,

	/// How many times a value already on the queue has been moved to a
	/// different position in the queue
	reorders: usize,
}
impl ReportData
{
	fn new() -> Self
	{
		Self {
			ready_consumed: 0,
			ready_consumed_bytes: 0,
			added: 0,
			added_bytes: 0,
			discarded_non_ready_reads: 0,
			added_read_bytes: 0,
			reorders: 0,
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

	report: ReportData,
}

impl OperandQueue
{
	pub fn new(ready_ops: impl Iterator<Item = Value>) -> Self
	{
		Self {
			stack: VecDeque::new(),
			queue: VecDeque::new(),
			ready: VecDeque::from_iter(ready_ops.map(|v| Operand::Val(v))),
			report: ReportData::new(),
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
	pub fn ready_iter<'a>(
		&'a mut self,
		mem: &'a mut Memory,
	) -> impl 'a + Iterator<Item = (Value, Option<(MemError, usize)>)>
	{
		struct RemoveIter<'b>
		{
			queue: &'b mut OperandQueue,
			mem: &'b mut Memory,
		}
		impl<'b> Iterator for RemoveIter<'b>
		{
			type Item = (Value, Option<(MemError, usize)>);

			fn next(&mut self) -> Option<Self::Item>
			{
				self.queue.ready.pop_front().and_then(|op| {
					let mut err = None;
					let val = match op
					{
						Operand::Val(v) => v,
						Operand::Read(op) =>
						{
							RefMut::map((*op).borrow_mut(), |op| {
								match op
								{
									Ok(v) => v,
									Err((addr, len, typ)) =>
									{
										let mut v = Value::new(*typ);
										err = self.mem.read_data(*addr, &mut v, *len).err();
										*op = Ok(v);
										op.as_mut().ok().unwrap()
									},
								}
							})
							.clone()
						},
					};

					self.queue.report.ready_consumed += 1;
					self.queue.report.ready_consumed_bytes += val.size();
					Some((val, err))
				})
			}
		}
		impl<'b> Drop for RemoveIter<'b>
		{
			fn drop(&mut self)
			{
				self.queue.report.discarded_non_ready_reads +=
					self.queue.ready.iter().fold(0, |mut acc, op| {
						if op.is_non_ready_read()
						{
							acc += 1;
						}
						acc
					});
				self.queue.ready = self.queue.queue.pop_front().unwrap_or(VecDeque::new());
			}
		}
		RemoveIter { queue: self, mem }
	}

	/// Pushes the given operand to the back of the queue at the given index.
	///
	/// Doesn't update any report data
	fn push_op_unreported(&mut self, idx: usize, op: Operand)
	{
		// Ensures the operand queue is long enough to include the given index.
		if self.queue.len() <= idx
		{
			self.queue.resize_with(idx + 1, || VecDeque::new());
		}
		self.queue[idx].push_back(op);
	}

	/// Pushes the given operand to the back of the queue at the given index.
	pub fn push_op(&mut self, idx: usize, op: Operand)
	{
		self.report.added += 1;
		if let Operand::Val(v) = &op
		{
			self.report.added_bytes += v.size()
		}
		self.push_op_unreported(idx, op);
	}

	/// Pushes the given value to the back of the queue at the given index.
	pub fn push_val(&mut self, idx: usize, v: Value)
	{
		self.push_op(idx, Operand::Val(v));
	}

	/// Pushes the given read type from the given address onto the queue at the
	/// given idx
	pub fn push_read(&mut self, idx: usize, addr: usize, typ: ValueType, len: usize)
	{
		self.push_op(
			idx,
			Operand::Read(Rc::new(RefCell::new(Err((addr, len, typ))))),
		);
	}

	/// Moves the operand found on the queue with index `src_q_idx`, position
	/// `src_idx` in that queue, to the back of the queue with idx 'target_idx'
	pub fn reorder(&mut self, src_q_idx: usize, src_idx: usize, target_idx: usize)
	{
		let op = self
			.queue
			.get_mut(src_q_idx)
			.and_then(|q| q.remove(src_idx))
			.map(|op| {
				self.report.reorders += 1;
				op
			})
			.unwrap_or(Operand::Val(Value::new_nar::<u8>(0)));
		self.push_op_unreported(target_idx, op);
	}

	/// Moves all operands on `src_idx` queue to the back of the `dest_idx`
	/// queue. If `src_idx`
	pub fn reorder_ready(&mut self, dest_idx: usize)
	{
		while let Some(op) = self.ready.remove(0).map(|op| {
			self.report.reorders += 1;
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
		self.report.discarded_non_ready_reads +=
			self.queue
				.iter()
				.flat_map(|qs| qs.iter())
				.fold(0, |mut acc, op| {
					if op.is_non_ready_read()
					{
						acc += 1;
					}
					acc
				});
		self.queue = self.stack.pop_back().unwrap_or(VecDeque::new());
	}
}
