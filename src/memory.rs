use crate::report::{Report, Reporter};
use num_traits::{Bounded, FromPrimitive, One, Unsigned, Zero};
use std::{net::Shutdown::Write, ops::Add};

/// A trait for types usable as memory addresses
pub trait Address: Sized + Copy + Bounded + Zero + One + Unsigned + FromPrimitive
{
	/// Returns whether the given address is aligned to the
	/// given power of 2 plus 1.
	/// I.e. `x.is_aligned(0)` returns whether x is aligned to 2^1=2,
	/// `x.is_aligned(1)` returns whether x is aligned to 4, etc.
	fn is_aligned(self, pow2: i8) -> bool
	{
		(self % FromPrimitive::from_i8(pow2).unwrap()) == Self::zero()
	}
}

/// The memory report data
struct ReportData
{
	/// The number of instruction bytes that were requested
	/// divided by 2.
	instr_read: u16,

	// The number of data bytes requested
	data_read: u16,

	// The number of data written
	data_write: u16,

	// The number of data requests that weren't aligned to the native word size
	unaligned_read: u16,

	// The number of writes that weren't aligned to the native word size
	unaligned_write: u16,
}
impl ReportData
{
	fn new() -> Self
	{
		Self {
			instr_read: 0,
			data_read: 0,
			data_write: 0,
			unaligned_read: 0,
			unaligned_write: 0,
		}
	}
}

struct Memory<A: Address>
{
	/// We track memory blocks that have been written to at least once.
	mem_blocks: Vec<(A, Vec<u8>)>,
}

pub struct MemoryInstr<A: Address>
{
	memory: Memory<A>,
	report: ReportData,
}
impl<A: Address> MemoryInstr<A>
{
	pub fn read_intr(&mut self, addr: A) -> &[u8; 2]
	{
		self.report.instr_read += 1;
		todo!()
	}

	pub fn skip(self) -> MemoryRead<A>
	{
		MemoryRead {
			memory: self.memory,
			report: self.report,
		}
	}
}
impl<A: Address> Reporter for MemoryInstr<A>
{
	type Args = ();
	type Report = MemoryReport<A>;

	fn step(self, _: Self::Args) -> Self::Report
	{
		MemoryReport {
			memory: self.memory,
			report: self.report,
		}
	}
}

pub struct MemoryRead<A: Address>
{
	memory: Memory<A>,
	report: ReportData,
}
impl<A: Address> MemoryRead<A>
{
	fn read_data(&mut self, addr: A, bytes: u16) -> &[u8]
	{
		self.report.data_read += bytes;
		todo!()
	}

	fn write(self, addr: A, data: &[u8]) -> MemoryWrite<A>
	{
		let mut write = MemoryWrite {
			memory: self.memory,
			report: self.report,
		};
		write.write(addr, data);
		write
	}
}
impl<A: Address> Reporter for MemoryRead<A>
{
	type Args = ();
	type Report = MemoryReport<A>;

	fn step(self, _: Self::Args) -> Self::Report
	{
		MemoryReport {
			memory: self.memory,
			report: self.report,
		}
	}
}

pub struct MemoryWrite<A: Address>
{
	memory: Memory<A>,
	report: ReportData,
}
impl<A: Address> MemoryWrite<A>
{
	fn write(&mut self, addr: A, data: &[u8])
	{
		self.report.data_write += data.len() as u16;
		todo!()
	}
}
impl<A: Address> Reporter for MemoryWrite<A>
{
	type Args = ();
	type Report = MemoryReport<A>;

	fn step(self, _: Self::Args) -> Self::Report
	{
		MemoryReport {
			memory: self.memory,
			report: self.report,
		}
	}
}

pub struct MemoryReport<A: Address>
{
	memory: Memory<A>,
	report: ReportData,
}

impl<A: Address> Report for MemoryReport<A>
{
	type Reporter = MemoryInstr<A>;

	fn reset(self) -> Self::Reporter
	{
		MemoryInstr {
			memory: self.memory,
			report: ReportData::new(),
		}
	}
}
