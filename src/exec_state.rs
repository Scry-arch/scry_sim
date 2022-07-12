use crate::{Value, ValueType};
use std::{collections::HashMap, fmt::Debug, iter::once};

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
	pub fn extract_value(self) -> Value
	{
		match self
		{
			Self::Ready(v) => v,
			Self::MustRead(_) => panic!("OperandState is not ready"),
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

/// A functions information about return address, currently issued control flow,
/// and operand queues.
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct CallFrameState
{
	/// Where execution should continue after a return instruction
	pub ret_addr: usize,

	/// Issued branches that have yet to trigger,
	/// Key is the address after which the branch should trigger.
	/// Value is the type of control flow to trigger.
	pub branches: HashMap<usize, ControlFlowType>,

	/// Operands queues, each with its index.
	/// A missing index means that queue is empty.
	/// In the value, the first element is the first operand. The vec is then
	/// the rest of the operands, ordered.
	///
	/// Operands that are awaiting a read have the index of their read
	/// in `reads`.
	pub op_queues: HashMap<usize, (OperandState<usize>, Vec<OperandState<usize>>)>,

	/// Read instructions that have been issued but still waiting.
	///
	/// First element is the address to read from.
	/// Second is the number of scalars to read.
	/// Third is the type of scalars to read.
	pub reads: Vec<(usize, usize, ValueType)>,
}
impl CallFrameState
{
	/// Returns the number of times the given index is referenced by a MustRead
	/// operand in the given map
	pub(crate) fn count_read_refs(&self, idx: usize) -> usize
	{
		self.op_queues.iter().fold(0, |c, (_, (f, q))| {
			once(f).chain(q.iter()).fold(c, |mut c, op| {
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
	/// * Any operand queue has more than supported number of operands (4)
	pub fn valid(&self) -> bool
	{
		let unreferenced = (0..self.reads.len()).any(|idx| self.count_read_refs(idx) == 0);

		let read_nothing = self.reads.iter().any(|(_, len, _)| *len == 0);

		let wrong_ref = self.op_queues.iter().any(|(_, (op1, op_rest))| {
			for op in once(op1).chain(op_rest.iter())
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

		let too_many_operands = self
			.op_queues
			.iter()
			.any(|(_, (_, op_rest))| (op_rest.len()) > 3);

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
			self.op_queues.iter_mut().for_each(|(_, (f, q))| {
				once(f).chain(q.iter_mut()).for_each(|op| {
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
			op_queues: Default::default(),
			reads: Default::default(),
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
