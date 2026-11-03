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
	Produce(scry_isa::Type),
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

/// Specified which part of the stack has changed address
#[non_exhaustive]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum StackEvent
{
	Base,
	Split,
	Top,
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

	/// Change of the position of the stack (base/split/stack, new address)
	Stack(StackEvent, usize),

	/// An event connected to and ID (ID, event)
	Id(usize, IdEvent),
}

/// Track simulation events.
///
/// Implementations can choose to ignore some events (or all, as in the case of
/// `()`).
pub trait EventTracker
{
	/// Returns a new ID for use in an `IdEvent`.
	///
	/// It is implementation-defined whether the IDs are unique.
	/// The given ID will be used in creating an `IdEvent`.
	/// If uniquely identifying IdEvents is required, the implementator must
	/// ensure only unique IDs are returned.
	fn new_id(&mut self) -> usize;

	/// Reports an event.
	///
	/// The events are reported without a meaningful order.
	fn event(&mut self, event: Event);
}

/// Placeholder event tracker that doesn't track anything, for when no
/// tracking is needed.
impl EventTracker for ()
{
	fn new_id(&mut self) -> usize
	{
		0
	}

	fn event(&mut self, _: Event) {}
}
