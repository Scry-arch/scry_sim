use crate::Value;
use std::{
	collections::HashMap,
	fmt::Debug,
	iter::{once, Chain, Once},
	mem::replace,
	vec::IntoIter,
};

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
	pub first: Value,

	/// The rest of the operands in the list. In order.
	pub rest: Vec<Value>,
}
impl IntoIterator for OperandList
{
	type IntoIter = Chain<Once<Value>, IntoIter<Value>>;
	type Item = Value;

	fn into_iter(self) -> Self::IntoIter
	{
		once(self.first).chain(self.rest)
	}
}
impl OperandList
{
	/// Construct new operand list with the given first operand and the rest
	pub fn new(first: Value, rest: Vec<Value>) -> Self
	{
		Self { first, rest }
	}

	/// Iterates over the operands in-order
	pub fn iter(&self) -> impl Iterator<Item = &Value>
	{
		once(&self.first).chain(self.rest.iter())
	}

	/// Iterates over the operands in-order
	pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Value>
	{
		once(&mut self.first).chain(self.rest.iter_mut())
	}

	/// Adds an operand to the end of the list
	pub fn push(&mut self, op: Value)
	{
		self.rest.push(op)
	}

	/// Adds an operand to the front of the list
	pub fn push_first(&mut self, val: Value)
	{
		self.rest.insert(0, replace(&mut self.first, val));
	}

	/// Gets the number of operands in the list
	pub fn len(&self) -> usize
	{
		1 + self.rest.len()
	}

	/// Adds all the given operands to the end of the list
	pub fn extend(&mut self, iter: impl Iterator<Item = Value>)
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
	/// Memory block comprising the stack.
	pub block: Block,

	/// The size of the primary stack frame in bytes.
	pub base_size: usize,
}
impl StackFrame
{
	/// If the stack frame is valid, returns Ok, otherwise returns an err with a
	/// message
	pub fn validate(&self) -> Result<(), &str>
	{
		if self.block.size < self.base_size
		{
			return Err("Primary stack frame is larger than the whole stack frame");
		}

		if self.block.address.checked_add(self.block.size).is_none()
		{
			return Err("Block size overflows address space");
		}

		Ok(())
	}

	/// The size of the secondary stack frame.
	pub fn secondary_size(&self) -> usize
	{
		self.block.size - self.base_size
	}

	/// Returns whether the primary stack frame can be increased by the given
	/// amount in bytes
	///
	/// Increasing the primary frame decreases the secondary frame by an equal
	/// amount.
	pub fn can_increase_primary(&self, amount: usize) -> bool
	{
		(self.base_size + amount) >= self.block.size
	}

	/// Increases the size of the primary frame by the given amount.
	///
	/// Increasing the primary frame decreases the secondary frame by an equal
	/// amount.
	pub fn increase_primary(&mut self, amount: usize)
	{
		assert!(self.can_increase_primary(amount));
		self.base_size += amount;
	}

	/// Increases the size of the primary frame by the given amount.
	///
	/// Increasing the primary frame decreases the secondary frame by an equal
	/// amount.
	pub fn decrease_primary(&mut self, amount: usize)
	{
		assert!(
			amount <= self.base_size,
			"Cannot decrease primary stack by more than its size."
		);
		self.base_size += amount;
	}

	/// Increases the size of the stack.
	///
	/// If `primary=true`, then the added size is reserved to the primary stack.
	/// Otherwise, to the secondary.
	pub fn add_block(&mut self, amount: usize, primary: bool)
	{
		assert!(amount > 0);
		self.block.size += amount;

		if primary
		{
			self.base_size += amount;
		}
	}

	/// Releases the given number of bytes from the stack frame.
	///
	/// If the resulting total stack size is less than the primary stack size
	/// (before the call), then the primary size is decreased to fit.
	pub fn release_bytes(&mut self, amount: usize)
	{
		assert!(amount > 0);
		assert!(self.block.size >= amount, "Releasing more than frame size.");

		self.block.size -= amount;

		if self.base_size < self.block.size
		{
			self.base_size = self.block.size;
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
		let base_offset = if primary_base { 0 } else { self.base_size };

		let scalar_size = 1 << scalar_pow2;
		let frame_offset = base_offset + (index * scalar_size);

		if frame_offset >= self.block.size
		{
			None
		}
		else
		{
			Some(self.block.address + frame_offset)
		}
	}

	/// Returns the base address of the stack frame
	pub fn get_base_addres(&self) -> usize
	{
		self.get_address(0, 0, true).unwrap()
	}

	/// Returns the stack frame that would result from performing a call from
	/// this stack frame.
	pub fn frame_call(&mut self) -> StackFrame
	{
		let new_stack_base = self.block.address + self.base_size;
		let new_block = Self {
			block: Block {
				address: new_stack_base,
				size: self.secondary_size(),
			},
			base_size: self.secondary_size(),
		};

		self.block.size -= self.secondary_size();
		new_block
	}

	/// Updates this stack frame based on a return from the given stack frame.
	pub fn frame_return(&mut self, callee_frame: StackFrame)
	{
		self.block.size += callee_frame.base_size;
	}
}

/// References a block of memory with its base address and the size in bytes.
#[derive(Debug, Clone, Eq, PartialEq, Default)]
pub struct Block
{
	pub address: usize,
	pub size: usize,
}
impl Block
{
	pub fn validate<const ALLOW_EMPTY: bool>(&self) -> Result<(), String>
	{
		if self.address.checked_add(self.size).is_none()
		{
			return Err("Block size overflow".to_string());
		}

		if self.size == 0 && !ALLOW_EMPTY
		{
			return Err("Block size is 0".to_string());
		}
		Ok(())
	}

	/// Returns whether the given address is in this block.
	pub fn address_in(&self, address: usize) -> bool
	{
		(self.address <= address) && (address < self.address + self.size)
	}

	/// Returns whether the given block overlaps this one.
	///
	/// I.e. if any of the other addresses referenced by the other is in this.
	/// Not the other way.
	pub fn overlapped_by(&self, other: &Block) -> bool
	{
		self.address_in(other.address) || self.address_in(other.address + other.size - 1)
	}

	/// Returns whether the given block is fully within this one
	pub fn encompasses(&self, other: &Block) -> bool
	{
		self.address_in(other.address) && self.address_in(other.address + other.size - 1)
	}

	/// Returns whether the two block overlap
	pub fn overlap(block1: &Block, block2: &Block) -> bool
	{
		block1.overlapped_by(block2) || block2.overlapped_by(block1)
	}

	/// Return whether any of the given block overlap with eack other
	pub fn any_overlaps<'a>(
		blocks: impl Iterator<Item = &'a Block> + Clone,
	) -> Option<(&'a Block, &'a Block)>
	{
		for (i, block) in blocks.clone().enumerate()
		{
			if let Some(other) = blocks
				.clone()
				.skip(i + 1)
				.find(|b| Block::overlap(block, b))
			{
				return Some((block, other));
			}
		}
		None
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

	/// Stack frame
	pub stack: StackFrame,
}
impl CallFrameState
{
	/// Returns whether this call frame is valid.
	///
	/// It is invalid if:
	/// * Return address is not 2-byte aligned and within given address space
	/// * Any control flow trigger address is not 2-byte aligned and within
	///   given address space
	/// * Any control flow target address is not 2-byte aligned and within given
	///   address space
	/// * Any operand list has more than supported number of operands (4)
	pub fn validate(&self, max_addr: usize) -> Result<(), String>
	{
		if self.ret_addr % 2 != 0
		{
			return Err("Unaligned return address".to_string());
		}
		if self.ret_addr > max_addr
		{
			return Err("Return address overflows address space".to_string());
		}

		if self.branches.iter().any(|(trig, ctrl)| {
			trig % 2 != 0
				|| match ctrl
				{
					ControlFlowType::Call(targ) | ControlFlowType::Branch(targ) => targ % 2 != 0,
					_ => false,
				}
		})
		{
			return Err("Control flow target address unaligned".to_string());
		}
		if self.branches.iter().any(|(trig, ctrl)| {
			trig > &max_addr
				|| match ctrl
				{
					ControlFlowType::Call(targ) | ControlFlowType::Branch(targ) => targ > &max_addr,
					_ => false,
				}
		})
		{
			return Err("Control flow target address overflows address space".to_string());
		}

		if self.op_queue.iter().any(|(_, ops)| ops.len() > 4)
		{
			return Err("Too many operands in queue".to_string());
		}

		Ok(())
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
	/// The size of the memory space in powers of two bytes.
	///
	/// I.e., when 0, 1 byte is enough to address the whole space,
	/// when 1, 2 bytes is enough, etc.
	pub addr_space: u8,

	/// Next address to be executed.
	pub address: usize,

	/// Current call frame
	pub frame: CallFrameState,

	/// Call frames of callers of the current function.
	///
	/// The first frame is of the current function's caller, the next is the
	/// caller of the caller and so on. May be empty.
	pub frame_stack: Vec<CallFrameState>,

	/// Number of free bytes available for the program stack.
	///
	/// The base address of the whole stack is given in the base stack frame.
	pub stack_buffer: usize,
}
impl ExecState
{
	/// Whether this state is valid.
	///
	/// It is invalid if:
	/// * Any call frame is invalid
	/// * Next address is not 2-byte aligned or overflows address space
	/// * The address space is no large than usize
	pub fn validate(&self) -> Result<(), String>
	{
		let frames = once(&self.frame).chain(self.frame_stack.iter());

		// Validate frames
		for f in frames.clone()
		{
			if let Err(msg) = f.validate(self.max_addr())
			{
				return Err(format!("Invalid call frame: {}", msg));
			}
		}

		// Check no stack blocks overlap
		self.frame_stack.iter().fold(
			Ok(self.frame.stack.block.address + self.frame.stack.block.size),
			|addr: Result<_, String>, frame| {
				if let Ok(addr) = addr
				{
					if addr != frame.stack.block.address
					{
						return Err("Non contiguous stack frame addresses".into());
					}
					else
					{
						Ok(frame.stack.block.address + frame.stack.block.size)
					}
				}
				else
				{
					addr
				}
			},
		)?;

		// Check address alignment
		if self.address % 2 != 0
		{
			return Err("Unaligned next address".into());
		}

		if size_of::<usize>() < self.pointer_size() as usize
		{
			return Err("Oversized address space".into());
		}
		if self.address > self.max_addr()
		{
			return Err("Next instruction overflowing address space".into());
		}

		Ok(())
	}

	/// Returns the address space size in bytes.
	///
	/// This is the smallest pointer size that can address the whole memory
	/// address space.
	pub fn pointer_size(&self) -> u8
	{
		2u8.pow(self.addr_space as u32)
	}

	/// Returns the highest address in the address space.
	pub fn max_addr(&self) -> usize
	{
		2usize.saturating_pow((self.pointer_size() * 8) as u32)
	}
}
