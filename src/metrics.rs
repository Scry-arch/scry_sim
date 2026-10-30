use scry_isa::Instruction;
use std::collections::HashMap;
use strum::{EnumIter, IntoEnumIterator};

/// Various metrics about a given simulation run.
///
/// All are assumed to be non-negative integers with the initial value of 0.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, EnumIter)]
pub enum Metric
{
	// Control flow
	/// Branching control flow issued
	IssuedBranches,
	/// Function calling control flow issued
	IssuedCalls,
	/// Function returning control flow issued
	IssuedReturns,
	/// Branching control flow triggered
	TriggeredBranches,
	/// Function calling control flow triggered
	TriggeredCalls,
	/// Function returning control flow triggered
	TriggeredReturns,

	// Operands
	/// Operands consumed by instructions
	ConsumedOperands,
	/// Operand bytes consumed by instructions
	ConsumedBytes,
	/// Value operands queued on any operand list
	QueuedValues,
	/// Value operand bytes queued on any operand list
	QueuedValueBytes,
	/// Operands reordered from one operand list into another
	ReorderedOperands,

	// Stack
	/// Total number of total stack frame reserves executed
	StackReserveTotal,
	/// Total bytes reserved for the total stack frame.
	///
	/// A base stack frame reserve that also increases the total stack frame
	/// will reflect the increased bytes in this metric.
	StackReserveTotalBytes,
	/// Total number of base stack frame reserves executed
	StackReserveBase,
	/// Total bytes reserved for the base stack frame
	StackReserveBaseBytes,
	/// Total number of total stack frame frees executed
	StackFreeTotal,
	/// Total bytes freed from the total stack frame
	StackFreeTotalBytes,
	/// Total number of base stack frame frees executed
	StackFreeBase,
	/// Total bytes freed from the base stack frame.
	///
	/// Freeing of total stack frame that forces the base frame to be freed will
	/// reflect in this metric
	StackFreeBaseBytes,
	/// Data reads from stack performed
	StackReads,
	/// Data bytes read from stack
	StackReadBytes,
	/// Data writes to stack
	StackWrites,
	/// Data bytes written to stack
	StackWriteBytes,

	// Memory
	/// Instructions read from memory
	InstructionReads,
	/// Data reads from memory performed
	DataReads,
	/// Data bytes read from memory
	DataReadBytes,
	/// Data bytes written to memory
	DataWriteBytes,
	/// Unaligned data reads
	///
	/// Only counted once per vector read.
	UnalignedReads,
	/// Unaligned data writes
	///
	/// Only counted once per vector written.
	UnalignedWrites,
}
impl Metric
{
	/// Return an iterator over all the metrics
	pub fn all() -> impl Iterator<Item = Self>
	{
		Self::iter()
	}
}
/// Track simulation metrics.
///
/// Not all metrics are necessarily tracked.
/// Implementations can choose to ignore some metrics (or all, as in the case of
/// `()`).
pub trait MetricTracker
{
	/// Add the given amount to the current tracked amount of the given metric.
	///
	/// If the given metric is not tracked, does nothing.
	fn add_stat(&mut self, metric: Metric, amount: usize);
}

/// Report simulation metrics
///
/// Metrics all initialize to zero.
pub trait MetricReporter
{
	/// Returns the current count for the given metric.
	fn get_stat(&self, stat: Metric) -> usize;
}

/// Placeholder metric tracker that doesn't track anything, for when no
/// tracking/reporting is needed.
///
/// Note that `()` does not implement `MetricReporter` on purpose, so as not to
/// accidentally use it where a proper tracker/reporter should be used.
impl MetricTracker for ()
{
	fn add_stat(&mut self, _: Metric, _: usize) {}
}

/// A HashMap based metric tracker/reporter.
///
/// Tracks everything and can be compared to itself, resulting in true if all
/// metrics are the same.
#[derive(Debug, Clone, Eq)]
pub struct TrackReport
{
	stats: HashMap<Metric, usize>,
}
impl TrackReport
{
	pub fn new() -> Self
	{
		Self {
			stats: HashMap::new(),
		}
	}

	pub fn reset_stat(&mut self, stat: Metric)
	{
		self.stats.remove(&stat);
	}
}
impl MetricTracker for TrackReport
{
	fn add_stat(&mut self, stat: Metric, amount: usize)
	{
		if let Some(m) = self.stats.get_mut(&stat)
		{
			*m += amount;
		}
		else
		{
			self.stats.insert(stat, amount);
		}
	}
}
impl MetricReporter for TrackReport
{
	fn get_stat(&self, stat: Metric) -> usize
	{
		if let Some(m) = self.stats.get(&stat)
		{
			*m
		}
		else
		{
			0
		}
	}
}
impl<const LEN: usize> From<[(Metric, usize); LEN]> for TrackReport
{
	fn from(metrics: [(Metric, usize); LEN]) -> Self
	{
		Self {
			stats: metrics.into_iter().collect(),
		}
	}
}
/// Compares all metrics, returning equal if all metrics are equal, false
/// otherwise.
impl PartialEq for TrackReport
{
	fn eq(&self, other: &Self) -> bool
	{
		let right_eq_left = |r: &Self, l: &Self| {
			for (metric, count) in &r.stats
			{
				if l.get_stat(*metric) != *count
				{
					return false;
				}
			}
			true
		};
		right_eq_left(self, other) && right_eq_left(other, self)
	}
}

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

pub trait EventTracker
{
	fn new_operand_id(&mut self) -> usize;

	fn new_control_id(&mut self) -> usize;

	fn event(&mut self, event: Event);
}

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
