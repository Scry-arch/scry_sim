use scry_isa::Instruction;

/// Events connected to individual operands.
///
/// Operands can be produced or consumed by instructions.
/// They can be discarded in various ways or be forwarded to other instructions.
#[non_exhaustive]
#[derive(PartialEq, Eq, Clone, Debug, Copy, Hash)]
pub enum OperandEvent
{
	Consume,
	Produce,
	Discard,
	Forward,
}

/// Execution control event.
#[non_exhaustive]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum ControlEvent
{
	/// Simple address redirect
	Jump,

	/// Function call
	Call,

	/// Function return
	Return,

	/// Trap call
	TrapCall,

	/// Trap return
	TrapReturn,
}

/// An event bearing and identification token
#[non_exhaustive]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum IdEvent
{
	/// Operand related event
	Operand(OperandEvent),

	/// A control event is issued
	ControlIssue(ControlEvent),

	/// A control event is triggered
	ControlTrigger(ControlEvent),
}

/// Simulation events.
///
/// Simulation is divided into steps.
/// Events within a step are not ordered temporally, even though they might be
/// output in a predictable pattern or real-life would have event in some
/// specific order.
///
/// Some event type may happen multiple times per event.
#[non_exhaustive]
#[derive(PartialEq, Eq, Clone, Debug)]
pub enum Event
{
	/// Executed instruction (address, instruction)
	Instruction(usize, Instruction),

	/// Bytes loaded from memory (Effective address, Stack index, number of
	/// bytes)
	Load(usize, Option<usize>, u8),

	/// Bytes stored to memory (Effective address, Stack index, number of bytes)
	Store(usize, Option<usize>, u8),

	/// Change of the position of the stack (The split?, new address)
	Stack(bool, usize),

	/// An event connected to and ID (ID, event)
	Id(usize, IdEvent),
}

/// Track simulation events.
///
/// Implementations can choose to ignore some events (or all, as in the case of
/// `()`).
pub trait EventTracker
{
	fn new_operand_id(&mut self) -> usize;

	fn new_control_id(&mut self) -> usize;

	fn event(&mut self, event: Event);
}

/// Placeholder event tracker that doesn't track anything, for when no
/// tracking is needed.
impl EventTracker for ()
{
	fn new_operand_id(&mut self) -> usize
	{
		0
	}

	fn new_control_id(&mut self) -> usize
	{
		0
	}

	fn event(&mut self, _: Event) {}
}
