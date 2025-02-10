use std::collections::HashMap;

/// Various metrics about a given simulation run.
///
/// All are assumed to be non-negative integers with the initial value of 0.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
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
	/// Read operands queued on any operand list.
	///
	/// Note that duplicating an issued read that has yet to resolve also
	/// increments this counter.
	QueuedReads,
	/// Operands reordered from one operand list into another
	ReorderedOperands,

	// Stack
	/// Total number of total stack frame reserves executed
	StackReserveTotal,
	/// Total bytes reserved for the total stack frame
	StackReserveTotalBytes,

	// Memory
	/// Instructions read from memory
	InstructionReads,
	/// Data reads from memory performed
	DataReads,
	/// Data bytes read from memory
	DataReadBytes,
	/// Data bytes written to memory
	DataBytesWritten,
	/// Unaligned data reads
	///
	/// Only counted once per vector read.
	UnalignedReads,
	/// Unaligned data writes
	///
	/// Only counted once per vector written.
	UnalignedWrites,
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
