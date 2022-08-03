use crate::{
	value::{Scalar, Value},
	Metric, MetricTracker,
};
use bitvec::vec::BitVec;

#[derive(Debug)]
pub enum MemError
{
	/// Memory being read has not been initialized
	Uninitialized,
	/// Tried to write a NaR value to memory
	NarWrite,
	/// The address being accesses is out of alignment with the data type being
	/// accessed
	UnalignedAddr,
	/// The range of addresses requested includes invalid ones
	InvalidAddr,
}

/// Trait for operations on memory.
pub trait Memory
{
	/// Read from the given address, into the given value, the given number of
	/// elements.
	///
	/// The given value doesn't have to have correct or valid scalars compared
	/// to the value type. The read value will be valid compared to the type and
	/// the given length.
	///
	/// Updates the report as a data read.
	fn read_data(
		&mut self,
		addr: usize,
		into: &mut Value,
		len: usize,
		tracker: &mut impl MetricTracker,
	) -> Result<(), (MemError, usize)>;

	/// Read 2 bytes from the given address into the slice.
	///
	/// Updates the report as an instruction read.
	fn read_instr(
		&mut self,
		addr: usize,
		tracker: &mut impl MetricTracker,
	) -> Result<[u8; 2], MemError>;

	/// Write to the given address from the given value.
	///
	/// Updates the report as data is written
	fn write(
		&mut self,
		addr: usize,
		from: &Value,
		tracker: &mut impl MetricTracker,
	) -> Result<(), (MemError, usize)>;
}

#[derive(Debug)]
struct MemBlock
{
	data: Vec<u8>,
	initialized: BitVec,
}

/// Constructs a block with the given size in bytes.
/// All bytes are assumed uninitialized
impl From<usize> for MemBlock
{
	fn from(size: usize) -> Self
	{
		Self {
			data: vec![0; size],
			initialized: BitVec::repeat(false, size),
		}
	}
}

/// Constructs a block with the given bytes.
/// All bytes are initialized
impl From<&[u8]> for MemBlock
{
	fn from(bytes: &[u8]) -> Self
	{
		Self {
			data: bytes.into(),
			initialized: BitVec::repeat(true, bytes.len()),
		}
	}
}

/// Constructs a block with the given bytes.
/// All bytes are initialized
impl From<Vec<u8>> for MemBlock
{
	fn from(bytes: Vec<u8>) -> Self
	{
		let len = bytes.len();
		Self {
			data: bytes,
			initialized: BitVec::repeat(true, len),
		}
	}
}

impl MemBlock
{
	/// Appends the given blocks memory at the end of this block's.
	#[allow(dead_code)]
	fn append<T>(&mut self, other: T)
	where
		MemBlock: From<T>,
	{
		let mut other_block: MemBlock = other.into();
		self.data.append(&mut other_block.data);
		self.initialized.append(&mut other_block.initialized);
	}

	/// Size of memory block in bytes
	fn size(&self) -> usize
	{
		assert_eq!(self.data.len(), self.initialized.len());
		self.data.len()
	}

	/// Try to read a given amount of bytes starting from the given address.
	fn read(&self, addr: usize, size: usize) -> Result<&[u8], (MemError, usize)>
	{
		let range = addr..(addr + size);
		self.data
			.get(range.clone())
			.map_or(Err((MemError::InvalidAddr, self.size())), |slice| {
				if let Some((uninit, _)) = self
					.initialized
					.get(range)
					.unwrap()
					.into_iter()
					.enumerate()
					.find(|(_, init)| !init.as_ref())
				{
					Err((MemError::Uninitialized, addr + uninit))
				}
				else
				{
					assert_eq!(slice.len(), size);
					Ok(slice)
				}
			})
	}

	fn write(&mut self, addr: usize, bytes: &[u8]) -> Result<(), (MemError, usize)>
	{
		let range = addr..(addr + bytes.len());
		let size = self.size();
		self.data
			.get_mut(range.clone())
			.map_or(Err((MemError::InvalidAddr, size)), |slice| {
				assert_eq!(bytes.len(), slice.len());
				slice
					.into_iter()
					.zip(bytes.into_iter())
					.for_each(|(mem, val)| {
						*mem = *val;
					});
				Ok(())
			})
			.and_then(|_| {
				self.initialized
					.get_mut(range)
					.unwrap()
					.into_iter()
					.for_each(|mut init| *init = true);
				Ok(())
			})
	}
}

/// Represents blocks of memory
#[derive(Debug)]
pub struct BlockedMemory
{
	/// Offsets + memory blocks
	/// Sorted by offset, biggest to smallest
	blocks: Vec<(usize, MemBlock)>,
}

impl BlockedMemory
{
	/// Construct a new memory object with only the given block
	pub fn new(mem: Vec<u8>, offset: usize) -> Self
	{
		Self {
			blocks: vec![(offset, mem.into())],
		}
	}

	/// Read from the given address, putting the read values into the given
	/// slice
	///
	/// This read does not update the read report in any way
	fn read_no_report(&self, addr: usize, size: usize) -> Result<&[u8], (MemError, usize)>
	{
		self.blocks
			.iter()
			.find(|(offset, _)| *offset <= addr)
			.ok_or((MemError::InvalidAddr, addr))
			.and_then(|(offset, mem)| mem.read(addr - offset, size))
	}
}
impl Memory for BlockedMemory
{
	/// Read from the given address, into the given value, the given number of
	/// elements
	///
	/// Updates the report as a data read.
	fn read_data(
		&mut self,
		addr: usize,
		into: &mut Value,
		_len: usize,
		tracker: &mut impl MetricTracker,
	) -> Result<(), (MemError, usize)>
	{
		let mut found_uninit = None;
		let scale = into.scale();
		let size = into.size();
		for (idx, val) in into.iter_mut().enumerate()
		{
			self.read_no_report(addr + (idx * scale), scale)
				.and_then(|bytes| {
					val.set_val(bytes);
					Ok(())
				})
				.and_then(|_| {
					tracker.add_stat(Metric::DataBytesRead, size);
					if addr % scale != 0
					{
						tracker.add_stat(Metric::UnalignedReads, 1);
					}
					Ok(())
				})
				.err()
				.map(|err| {
					if let (MemError::Uninitialized, addr) = err
					{
						found_uninit = Some(addr);
					}
					*val = Scalar::Nar(0);
				});
		}
		found_uninit.map_or(Ok(()), |addr| Err((MemError::Uninitialized, addr)))
	}

	/// Read 2 bytes from the given address into the slice.
	///
	/// Updates the report as an instruction read.
	fn read_instr(
		&mut self,
		addr: usize,
		tracker: &mut impl MetricTracker,
	) -> Result<[u8; 2], MemError>
	{
		if addr % 2 != 0
		{
			Err(MemError::UnalignedAddr)
		}
		else
		{
			self.read_no_report(addr, 2)
				.and_then(|bytes| Ok([bytes[0], bytes[1]]))
				.and_then(|bytes| {
					tracker.add_stat(Metric::InstructionReads, 1);
					Ok(bytes)
				})
				.or_else(|err| Err(err.0))
		}
	}

	fn write(
		&mut self,
		addr: usize,
		from: &Value,
		tracker: &mut impl MetricTracker,
	) -> Result<(), (MemError, usize)>
	{
		let (offset, mem) = self
			.blocks
			.iter_mut()
			.find(|(offset, _)| *offset < addr)
			.ok_or((MemError::InvalidAddr, addr))?;

		for (idx, val) in from.iter().enumerate()
		{
			let element_addr = (addr - *offset) + (idx * from.scale());
			if let Scalar::Val(bytes) = val
			{
				mem.write(element_addr, bytes.as_ref())?;
				tracker.add_stat(Metric::DataBytesWritten, from.size());
				if addr % from.scale() != 0
				{
					tracker.add_stat(Metric::DataBytesWritten, 1);
				}
			}
			else if let Scalar::Nar(_) = val
			{
				return Err((MemError::NarWrite, element_addr));
			}
			else
			{
				// For Nan, nothing should be written
			}
		}
		Ok(())
	}
}
