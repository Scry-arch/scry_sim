use crate::{
	value::{Scalar, Value},
	Metric, MetricTracker,
};
use bitvec::vec::BitVec;
use std::cmp::PartialEq;

#[derive(Debug, Eq, PartialEq)]
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
///
/// Required methods only need directly access to individual bytes, with
/// provided methods implementing accesses patterns and metric tracking.
pub trait Memory
{
	/// Gets the byte value at the given address (if a valid address).
	fn read_raw(&mut self, addr: usize) -> Result<u8, MemError>;

	/// Read from the given address, into the given value, the given number of
	/// elements.
	///
	/// The given value doesn't have to have correct or valid scalars compared
	/// to the value type. The read value will be valid compared to the type and
	/// the given length.
	///
	/// Updates the report as a data read.
	///
	/// If an error is thrown, the first element describes it and the second is
	/// the relevant address.
	#[allow(unused_variables)]
	fn read_data(
		&mut self,
		addr: usize,
		into: &mut Value,
		len: usize,
		tracker: &mut impl MetricTracker,
	) -> Result<(), (MemError, usize)>
	{
		let mut found_uninit = None;
		let scale = into.scale();
		'a: for (idx, val) in into.iter_mut().enumerate()
		{
			let start_addr = addr + (idx * scale);
			let mut bytes = Vec::with_capacity(scale);
			for b in 0..scale
			{
				match self.read_raw(start_addr + b)
				{
					Err(err) =>
					{
						if MemError::Uninitialized == err
						{
							found_uninit = Some(start_addr + b);
						}
						*val = Scalar::Nar(0);
						continue 'a;
					},
					Ok(byte) =>
					{
						bytes.push(byte);
					},
				}
			}

			val.set_val(bytes.as_slice());

			tracker.add_stat(Metric::DataReads, 1);
			tracker.add_stat(Metric::DataReadBytes, scale);
			if (addr % scale) != 0
			{
				tracker.add_stat(Metric::UnalignedReads, 1);
			}
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
			let bytes = [self.read_raw(addr)?, self.read_raw(addr + 1)?];
			tracker.add_stat(Metric::InstructionReads, 1);
			Ok(bytes)
		}
	}

	/// Sets the byte value at the given address to the given byte
	fn write_raw(&mut self, addr: usize, data: u8) -> Result<(), MemError>;

	/// Write to the given address from the given value.
	///
	/// Updates the report as data is written
	fn write(
		&mut self,
		addr: usize,
		from: &Value,
		tracker: &mut impl MetricTracker,
	) -> Result<(), (MemError, usize)>
	{
		for (idx, val) in from.iter().enumerate()
		{
			let element_addr = addr + (idx * from.scale());
			if let Scalar::Val(bytes) = val
			{
				for (i, b) in bytes.iter().enumerate()
				{
					self.write_raw(element_addr + i, *b)
						.map_err(|e| (e, element_addr))?;
				}

				tracker.add_stat(Metric::DataWriteBytes, bytes.len());
				if addr % from.scale() != 0
				{
					tracker.add_stat(Metric::UnalignedWrites, 1);
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
			.map_or(Err((MemError::InvalidAddr, addr)), |slice| {
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
	/// Constructs a new memory object with no blocks
	pub fn empty() -> Self
	{
		Self { blocks: vec![] }
	}

	/// Construct a new memory object with only the given block
	pub fn new(mem: impl Iterator<Item = u8>, offset: usize) -> Self
	{
		Self {
			blocks: vec![(offset, mem.collect::<Vec<_>>().into())],
		}
	}

	/// Asserts that the given block offset and size does not overlap with an
	/// existing block.
	///
	/// Causes an assertion error if overlapping.
	fn assert_non_overlapping(&self, offset: usize, size: usize)
	{
		let end_addr = offset + size;
		// Non-overlapping. TODO: handle overlapping
		assert!(self.blocks.iter().all(|(other_offset, other_block)| {
			let other_end_addr = other_offset + other_block.data.len();
			end_addr <= *other_offset || offset >= other_end_addr
		}));
	}

	/// Sort all the blocks by their offset, from low to high
	fn sort_blocks(&mut self)
	{
		self.blocks
			.sort_by(|(offset1, _), (offset2, _)| offset1.partial_cmp(offset2).unwrap())
	}

	/// Adds a new block to the memory at the given offset and with the given
	/// size.
	///
	/// All memory is zeroed and initialized
	pub fn add_block_zeroed(&mut self, offset: usize, size: usize)
	{
		self.assert_non_overlapping(offset, size);

		self.blocks.push((offset, MemBlock::from(size)));
		self.sort_blocks();
	}

	/// Adds a new block to the memory with the given data and offset
	pub fn add_block(&mut self, mem: impl Iterator<Item = u8>, offset: usize)
	{
		self.assert_non_overlapping(offset, mem.size_hint().0);

		self.blocks.push((offset, mem.collect::<Vec<_>>().into()));
		self.sort_blocks();
	}
}
impl Memory for BlockedMemory
{
	fn read_raw(&mut self, addr: usize) -> Result<u8, MemError>
	{
		self.blocks
			.iter()
			.rev()
			.find(|(offset, _)| *offset <= addr)
			.ok_or(MemError::InvalidAddr)
			.and_then(|(offset, mem)| mem.read(addr - offset, 1).map_err(|e| e.0))
			.map(|byte| byte[0])
	}

	fn write_raw(&mut self, addr: usize, data: u8) -> Result<(), MemError>
	{
		let (offset, mem) = self
			.blocks
			.iter_mut()
			.rev()
			.find(|(offset, _)| *offset <= addr)
			.ok_or((MemError::InvalidAddr, addr))
			.map_err(|err| err.0)?;

		let idx = addr - *offset;
		mem.write(idx, &[data]).map_err(|err| err.0)
	}
}

/// Used to access the host machine memory directly.
///
/// Has no guardrails, use with care.
/// Invalid accesses may cause crashes.
/// Has no synchronization between instances.
pub struct HostMemory();

impl Memory for HostMemory
{
	fn read_raw(&mut self, addr: usize) -> Result<u8, MemError>
	{
		let ptr = addr as *mut u8;
		Ok(unsafe { *ptr })
	}

	fn write_raw(&mut self, addr: usize, data: u8) -> Result<(), MemError>
	{
		let ptr = addr as *mut u8;
		unsafe { *ptr = data }
		Ok(())
	}
}
