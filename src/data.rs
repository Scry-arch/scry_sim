use crate::report::{Report, Reporter};
use std::collections::VecDeque;

/// Assumed to be integers for now (future might have different type, e.g.,
/// float)
struct ValueType
{
	// The scalar size of the integer
	size: u8,
	// Vector length
	length: u8,
}

impl ValueType
{
	fn bytes(&self) -> u16
	{
		(self.size as u16) * (self.length as u16)
	}
}

/// A piece of data.
struct Value
{
	typ: ValueType,
}

struct QueueData
{
	value_deque: VecDeque<Value>,
}
struct ReportData
{
	/// How many of the values at the head of the queue were consumed.
	consumed: u16,

	/// Where and what type of values were added to the queue.
	/// The list is ordered time-wise.
	added: Vec<(u16, ValueType)>,
}
impl ReportData
{
	fn new() -> Self
	{
		Self {
			consumed: 0,
			added: Vec::new(),
		}
	}
}

pub struct QueueCall
{
	queue: QueueData,
	report: ReportData,
}
impl QueueCall
{
	pub fn perform_call(self) -> QueueRead
	{
		todo!()
	}

	pub fn perform_return(self) -> QueueRead
	{
		todo!()
	}

	pub fn skip(self) -> QueueRead
	{
		todo!()
	}
}
impl Reporter for QueueCall
{
	type Args = ();
	type Report = QueueReport;

	fn step(self, args: Self::Args) -> Self::Report
	{
		todo!()
	}
}

/// The data queue
pub struct QueueRead
{
	queue: QueueData,
	report: ReportData,
}
impl QueueRead
{
	fn dequeue(&mut self) -> &Value
	{
		self.report.consumed += 1;
		todo!()
	}

	fn enqueue(self, v: Value, index: usize) -> QueueWrite
	{
		let mut write = QueueWrite {
			queue: self.queue,
			report: self.report,
		};
		write.enqueue(v, index);
		write
	}
}
impl Reporter for QueueRead
{
	type Args = ();
	type Report = QueueReport;

	fn step(self, args: Self::Args) -> Self::Report
	{
		todo!()
	}
}

/// The data queue
pub struct QueueWrite
{
	queue: QueueData,
	report: ReportData,
}
impl QueueWrite
{
	fn enqueue(&mut self, v: Value, index: usize)
	{
		todo!()
	}
}
impl Reporter for QueueWrite
{
	type Args = ();
	type Report = QueueReport;

	fn step(self, args: Self::Args) -> Self::Report
	{
		QueueReport {
			queue: self.queue,
			report: self.report,
		}
	}
}

/// The data store report after a step
pub struct QueueReport
{
	queue: QueueData,
	report: ReportData,
}
impl Report for QueueReport
{
	type Reporter = QueueRead;

	fn reset(self) -> Self::Reporter
	{
		QueueRead {
			queue: self.queue,
			report: ReportData::new(),
		}
	}
}
