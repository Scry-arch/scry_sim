use byteorder::{ByteOrder, LittleEndian};
use scry_sim::{MemError, Memory, Metric, MetricTracker, OperandQueue, Scalar, Value};
use std::mem::size_of;

/// A memory that always produces the same instruction and data.
///
/// All read operations will always succeed regardless of address or alignment.
///
/// The first member is the encoding of the instruction that the memory
/// should return from every `read_instr` while the second is what
/// `read_data` will populate all read bytes with.
///
/// Read metrics will be reported correctly, except alignment which will
/// be ignored.
///
/// If `MAY_STORE` is true, writes will succeed and update metric. If not,
/// writes will fail.
///
/// Meant for testing.
#[derive(Debug, Clone, Copy)]
pub struct RepeatingMem<const MAY_STORE: bool>(pub u16, pub u8);
impl<const MAY_STORE: bool> Memory for RepeatingMem<MAY_STORE>
{
	fn read_raw(&mut self, _: usize) -> Option<u8>
	{
		Some(self.1)
	}

	fn read_data(
		&mut self,
		_: usize,
		into: &mut Value,
		len: usize,
		tracker: &mut impl MetricTracker,
	) -> Result<(), (MemError, usize)>
	{
		let mut scalar_bytes = Vec::with_capacity(into.scale());
		scalar_bytes.resize(into.scale(), self.1);
		let scalar = Scalar::Val(scalar_bytes.into_boxed_slice());

		let mut scalars = Vec::with_capacity(len);
		scalars.resize_with(len, || scalar.clone());
		let result = Value::new_typed(into.value_type(), scalars.remove(0), scalars).unwrap();

		tracker.add_stat(Metric::DataBytesRead, result.size());

		*into = result;
		Ok(())
	}

	fn read_instr(
		&mut self,
		_: usize,
		tracker: &mut impl MetricTracker,
	) -> Result<[u8; 2], MemError>
	{
		let mut result = [0; 2];
		LittleEndian::write_u16(&mut result, self.0);
		tracker.add_stat(Metric::InstructionReads, 1);
		Ok(result)
	}

	fn write(
		&mut self,
		addr: usize,
		from: &Value,
		tracker: &mut impl MetricTracker,
	) -> Result<(), (MemError, usize)>
	{
		if MAY_STORE
		{
			tracker.add_stat(Metric::DataBytesWritten, from.size());
			Ok(())
		}
		else
		{
			Err((MemError::InvalidAddr, addr))
		}
	}
}

/// Advances the given operand queue by 1, i.e. all indices reduced by 1
pub fn advance_queue(from: OperandQueue) -> OperandQueue
{
	from.into_iter().map(|(idx, ops)| (idx - 1, ops)).collect()
}

/// Regress the given operand queue by 1, i.e. all indices increased by 1
pub fn regress_queue(from: OperandQueue) -> OperandQueue
{
	from.into_iter().map(|(idx, ops)| (idx + 1, ops)).collect()
}

/// Interprets the given scalar as a simulation address.
pub fn as_addr(operand: &Scalar) -> usize
{
	let mut result = 0usize;

	assert!(size_of::<usize>() >= operand.bytes().unwrap().len());

	for (idx, byte) in operand.bytes().unwrap().iter().enumerate()
	{
		let mut byte_usize = *byte as usize;
		let shift_count = idx * 8;
		byte_usize <<= shift_count;
		result += byte_usize;
	}

	result
}
