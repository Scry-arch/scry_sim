use crate::{Value, ValueType};
use std::{
	collections::HashMap,
	fmt::Debug,
	iter::{once, Chain, Once},
	vec::IntoIter,
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
///
/// `R` is the type used to refer to what needs reading
#[derive(Debug, Eq, PartialEq, Clone)]
pub enum OperandState<R>
{
	/// Operand is ready for use as the given value
	Ready(Value),

	/// Operand must read from memory to get the value needed.
	MustRead(R),
}
impl<R> OperandState<R>
{
	pub fn extract_value(self) -> Option<Value>
	{
		match self
		{
			Self::Ready(v) => Some(v),
			Self::MustRead(_) => None,
		}
	}

	pub fn get_value(&self) -> Option<&Value>
	{
		match self
		{
			Self::Ready(v) => Some(v),
			Self::MustRead(_) => None,
		}
	}
}

/// Control flow types
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum ControlFlowType
{
	/// A simple branch with target address.
	Branch(usize),
	/// A function call with the address of the function being called.
	Call(usize),
	/// A return from the current function being executed.
	Return,
}

/// Non-empty, ordered list of operands that an instruction will operate on.
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct OperandList
{
	/// First operand in the list
	pub first: OperandState<usize>,

	/// The rest of the operands in the list. In order.
	pub rest: Vec<OperandState<usize>>,
}
impl IntoIterator for OperandList
{
	type IntoIter = Chain<Once<OperandState<usize>>, IntoIter<OperandState<usize>>>;
	type Item = OperandState<usize>;

	fn into_iter(self) -> Self::IntoIter
	{
		once(self.first).chain(self.rest)
	}
}
impl OperandList
{
	/// Construct new operand list with the given first operand and the rest
	pub fn new(first: OperandState<usize>, rest: Vec<OperandState<usize>>) -> Self
	{
		Self { first, rest }
	}

	/// Iterates over the operands in-order
	pub fn iter(&self) -> impl Iterator<Item = &OperandState<usize>>
	{
		once(&self.first).chain(self.rest.iter())
	}

	/// Iterates over the operands in-order
	pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut OperandState<usize>>
	{
		once(&mut self.first).chain(self.rest.iter_mut())
	}

	/// Adds an operands to the end of the list
	pub fn push(&mut self, op: OperandState<usize>)
	{
		self.rest.push(op)
	}

	/// Gets the number of operands in the list
	pub fn len(&self) -> usize
	{
		1 + self.rest.len()
	}

	/// Adds all the given operands to the end of the list
	pub fn extend(&mut self, iter: impl Iterator<Item = OperandState<usize>>)
	{
		self.rest.extend(iter)
	}
}

/// The queue of operand lists that track what lists future instructions will
/// operate on.
///
/// Key 0 is the list the next instruction will receive. Key 1 is for the
/// instruction after that etc. If a key is not present, it means that operand
/// list has no operands. If that list is still empty when it reaches index 0,
/// that instruction will get no operands.
pub type OperandQueue = HashMap<usize, OperandList>;

/// Represents the state of the function stack frame.
///
/// The stack frame is not necessarily contiguous. Therefore, it consists of a
/// set of blocks in memory. We store eack block's base address and size in a
/// sorted list.
///
/// The stack frame is also divided into 2 parts: The primary and secondary
/// frames. We store the size of the primary frame, which may span multiple
/// blocks, while the size of the secondary frame would be what is left.
#[derive(Debug, Eq, PartialEq, Clone, Default)]
pub struct StackFrame
{
	/// Memory blocks comprising the stack.
	///
	/// 0. The address of the block.
	/// 0. The size of the block in bytes.
	///
	/// Lower index block also has lower index in the stack frame.
	/// [0].0 is therefore the base address of the frame.
	/// If empty, no memory is reserved for the stack frame.
	pub blocks: Vec<(usize, usize)>,

	/// The size of the primary stack frame in bytes.
	pub primary_size: usize,
}
impl StackFrame
{
	/// If the stack frame is valid, returns Ok, otherwise returns an err with a
	/// message
	pub fn validate(&self) -> Result<(), &str>
	{
		if self.blocks.iter().any(|(_, size)| *size == 0)
		{
			return Err("Stack frame block has size 0");
		}

		if self.total_size() < self.primary_size
		{
			return Err("Primary stack frame is larger than the whole stack frame");
		}

		if self
			.blocks
			.iter()
			.any(|(addr, size)| addr.checked_add(*size).is_none())
		{
			return Err("Block size overflows address space");
		}

		// Assert the block do not overlap
		fn block_overlaps_block(block1: (usize, usize), block2: (usize, usize)) -> bool
		{
			((block2.0 <= block1.0) && (block1.0 < (block2.0 + block2.1)))
				|| ((block2.0 < (block1.0 + block1.1))
					&& ((block1.0 + block1.1) <= (block2.0 + block2.1)))
		}

		for (idx, b) in self.blocks.iter().enumerate()
		{
			if self
				.blocks
				.iter()
				.skip(idx + 1)
				.any(|b2| block_overlaps_block(*b, *b2) || block_overlaps_block(*b2, *b))
			{
				return Err("Overlapping stack frame blocks");
			}
		}
		Ok(())
	}

	/// The total reserved size of the stack frame.
	pub fn total_size(&self) -> usize
	{
		self.blocks.iter().fold(0, |c, (_, size)| c + size)
	}

	/// The size of the secondary stack frame.
	pub fn secondary_size(&self) -> usize
	{
		self.total_size() - self.primary_size
	}

	/// Returns whether the primary stack frame can be increased by the given
	/// amount in bytes
	///
	/// Increasing the primary frame decreases the secondary frame by an equal
	/// amount.
	pub fn can_increase_primary(&self, amount: usize) -> bool
	{
		(self.primary_size + amount) >= self.total_size()
	}

	/// Increases the size of the primary frame by the given amount.
	///
	/// Increasing the primary frame decreases the secondary frame by an equal
	/// amount.
	pub fn increase_primary(&mut self, amount: usize)
	{
		assert!(self.can_increase_primary(amount));
		self.primary_size += amount;
	}

	/// Increases the size of the primary frame by the given amount.
	///
	/// Increasing the primary frame decreases the secondary frame by an equal
	/// amount.
	pub fn decrease_primary(&mut self, amount: usize)
	{
		assert!(
			amount <= self.primary_size,
			"Cannot decrease primary stack by more than its size."
		);
		self.primary_size += amount;
	}

	/// Adds the given block to the frame.
	///
	/// If `primary=true`, then the added size is reserved to the primary stack.
	/// Otherwise, to the secondary.
	pub fn add_block(&mut self, addr: usize, size: usize, primary: bool)
	{
		self.blocks.push((addr, size));

		if primary
		{
			self.primary_size += size;
		}
	}

	/// Releases the given number of bytes from the stack frame.
	///
	/// If the resulting total stack size is less than the primary stack size
	/// (before the call), then the primary size is decreased to fit.
	pub fn release_bytes(&mut self, amount: usize)
	{
		assert!(
			self.total_size() >= amount,
			"Releasing more than frame size."
		);

		let mut released = 0;

		while released < amount
		{
			let remaining = amount - released;

			if remaining >= self.blocks.last().unwrap().1
			{
				let popped_amount = self.blocks.pop().unwrap().1;
				released += popped_amount;
			}
			else
			{
				self.blocks.last_mut().unwrap().1 -= remaining;
			}
		}
		assert_eq!(released, amount);

		if self.primary_size < self.total_size()
		{
			self.primary_size = self.total_size();
		}
	}

	/// Returns the address corresponding to the given index of the given scalar
	/// size.
	///
	/// If the index is out of bounds, returns `None`.
	/// If `primary_base=true` uses the primary frame's base address as the
	/// starting point, otherwise the secondary frame's base.
	pub fn get_address(&self, scalar_pow2: u8, index: usize, primary_base: bool) -> Option<usize>
	{
		// We first find the relative address in the stack frame that we need (frame
		// offset)
		let base_offset = if primary_base { 0 } else { self.primary_size };

		let scalar_size = 1 << scalar_pow2;
		let frame_offset = base_offset + (index * scalar_size);

		// Now need to find the block containing that offset
		self.blocks
			.iter()
			.fold(Err(frame_offset), |offset, (addr, size)| {
				if let Err(offset) = offset
				{
					if *size < offset
					{
						Err(offset - size)
					}
					else
					{
						Ok(*addr)
					}
				}
				else
				{
					offset
				}
			})
			.ok()
	}

	/// Get the index of the block containing the split between the primary and
	/// secondary stack frames. Also returns the size of the __secondary__
	/// frame in that block.
	///
	/// If the secondary frame is empty, returns None.
	fn get_split(&self) -> Option<(usize, usize)>
	{
		self.blocks
			.iter()
			.enumerate()
			.rev()
			.fold(Err(self.secondary_size()), |remaining, (idx, (_, size))| {
				if let Err(remaining) = remaining
				{
					if remaining > *size
					{
						Err(remaining - size)
					}
					else
					{
						Ok((idx, remaining))
					}
				}
				else
				{
					remaining
				}
			})
			.ok()
	}

	/// Returns the stack frame that would result from performing a call from
	/// this stack frame.
	pub fn frame_call(&mut self) -> StackFrame
	{
		if self.secondary_size() > 0
		{
			let (block_idx, size) = self.get_split().unwrap();

			let first_block = self.blocks[block_idx];
			let mut new_blocks = vec![(first_block.0 + (first_block.1 - size), size)];

			self.blocks
				.iter()
				.skip(block_idx + 1)
				.for_each(|b| new_blocks.push(*b));

			Self {
				blocks: new_blocks,
				primary_size: self.secondary_size(),
			}
		}
		else
		{
			Self {
				blocks: vec![],
				primary_size: 0,
			}
		}
	}

	/// Updates this stack frame based on a return from the given stack frame.
	pub fn frame_return(&mut self, callee_frame: StackFrame)
	{
		// First, drop the secondary frame blocks
		if let Some((split_idx, secondary_size)) = self.get_split()
		{
			self.blocks.truncate(split_idx + 1);
			self.blocks.last_mut().unwrap().1 -= secondary_size;
		}

		// Now add the callee's primary frame blocks
		if callee_frame.primary_size > 0
		{
			assert!(!callee_frame.blocks.is_empty());
			let (split_idx, secondary_size) = callee_frame.get_split().unwrap_or((
				callee_frame.blocks.len() - 1,
				callee_frame.blocks.last().unwrap().1,
			));

			// Add all the full blocks
			callee_frame
				.blocks
				.iter()
				.take(split_idx)
				.for_each(|b| self.blocks.push(*b));
			// Add the potentially partial last block
			self.blocks.push((
				callee_frame.blocks[split_idx].0,
				callee_frame.blocks[split_idx].1 - secondary_size,
			));
		}
	}
}

/// A functions information about return address, currently issued control flow,
/// and operand queue.
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct CallFrameState
{
	/// Where execution should continue after a return instruction
	pub ret_addr: usize,

	/// Issued branches that have yet to trigger,
	/// Key is the address after which the branch should trigger.
	/// Value is the type of control flow to trigger.
	pub branches: HashMap<usize, ControlFlowType>,

	/// Operands queue
	pub op_queue: OperandQueue,

	/// Read instructions that have been issued but still waiting.
	///
	/// First element is the address to read from.
	/// Second is the number of scalars to read.
	/// Third is the type of scalars to read.
	pub reads: Vec<(usize, usize, ValueType)>,

	/// Stack frame
	pub stack: StackFrame,
}
impl CallFrameState
{
	/// Returns the number of times the given index is referenced by a MustRead
	/// operand in the given map
	pub(crate) fn count_read_refs(&self, idx: usize) -> usize
	{
		self.op_queue.iter().fold(0, |c, (_, ops)| {
			ops.iter().fold(c, |mut c, op| {
				if let OperandState::MustRead(read_idx) = op
				{
					if *read_idx == idx
					{
						c += 1;
					}
				}
				c
			})
		})
	}

	/// Returns whether this call frame is valid.
	///
	/// It is invalid if:
	/// * A MustRead operand references a read index that doesn't exist.
	/// * A read is not references by any MustRead operand.
	/// * Return address is not 2-byte aligned
	/// * Any control flow trigger address is not 2-byte aligned
	/// * Any control flow target address is not 2-byte aligned
	/// * Any operand list has more than supported number of operands (4)
	pub fn valid(&self) -> bool
	{
		let unreferenced = (0..self.reads.len()).any(|idx| self.count_read_refs(idx) == 0);

		let read_nothing = self.reads.iter().any(|(_, len, _)| *len == 0);

		let wrong_ref = self.op_queue.iter().any(|(_, ops)| {
			for op in ops.iter()
			{
				match op
				{
					OperandState::MustRead(idx) =>
					{
						if *idx >= self.reads.len()
						{
							return true;
						}
					},
					_ => (),
				}
			}
			false
		});

		let ret_addr_aligned = self.ret_addr % 2 == 0;

		let ctrl_unaligned = self.branches.iter().any(|(trig, ctrl)| {
			trig % 2 != 0
				|| match ctrl
				{
					ControlFlowType::Call(targ) | ControlFlowType::Branch(targ) => targ % 2 != 0,
					_ => false,
				}
		});

		let too_many_operands = self.op_queue.iter().any(|(_, ops)| ops.len() > 4);

		!unreferenced
			&& !wrong_ref
			&& ret_addr_aligned
			&& !ctrl_unaligned
			&& !too_many_operands
			&& !read_nothing
	}

	/// Remove the read with the given index if there is no references to it
	/// from MustRead operands. Returns whether any reads were removed.
	fn remove_read_if_unused(&mut self, idx: usize) -> bool
	{
		// Remove read with given index, if no operand references it
		if self.count_read_refs(idx) == 0
		{
			// Remove read
			self.reads.remove(idx);
			// Correct any references to higher-indexed reads
			self.op_queue.iter_mut().for_each(|(_, ops)| {
				ops.iter_mut().for_each(|op| {
					if let OperandState::MustRead(read_idx) = op
					{
						if *read_idx > idx
						{
							*read_idx -= 1;
						}
					}
				})
			});
			true
		}
		else
		{
			false
		}
	}

	/// Removes any reads that have no MustRead operands referencing them
	pub fn clean_reads(&mut self)
	{
		'outer: loop
		{
			for idx in 0..self.reads.len()
			{
				if self.remove_read_if_unused(idx)
				{
					continue 'outer;
				}
			}
			// All reads have references
			break;
		}
	}
}
impl Default for CallFrameState
{
	fn default() -> Self
	{
		Self {
			ret_addr: 0,
			branches: Default::default(),
			op_queue: Default::default(),
			reads: Default::default(),
			stack: Default::default(),
		}
	}
}

/// The specific state a Scry execution core is in.
///
/// An execution state contains all core information between two execution
/// steps. An execution step starts by executing an instruction and then checks
/// whether any control flow is triggered. If so, the control flow is then
/// executed before the step ends.
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct ExecState
{
	/// Next address to be executed.
	pub address: usize,

	/// Current call frame
	pub frame: CallFrameState,

	/// Call frames of callers of the current function.
	///
	/// The first frame is of the current function's caller, the next is the
	/// caller of the caller and so on. May be empty.
	pub frame_stack: Vec<CallFrameState>,
}
impl ExecState
{
	/// Whether this state is valid.
	///
	/// It is invalid if:
	/// * Any call frame is invalid
	/// * Next address is not 2-byte aligned
	pub fn valid(&self) -> bool
	{
		once(&self.frame)
			.chain(self.frame_stack.iter())
			.all(CallFrameState::valid)
			&& (self.address % 2 == 0)
	}

	/// Goes through all pending reads and ensures that they are referenced by
	/// an operand. Any read not referenced is removed.
	pub fn clean_reads(&mut self)
	{
		once(&mut self.frame)
			.chain(self.frame_stack.iter_mut())
			.for_each(CallFrameState::clean_reads);
	}
}
