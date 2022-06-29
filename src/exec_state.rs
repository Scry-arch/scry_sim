use crate::{Value, ValueType};
use std::{collections::HashMap, fmt::Debug};

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
