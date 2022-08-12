use byteorder::{ByteOrder, LittleEndian};
use scry_sim::{MemError, Memory, Metric, MetricTracker, Scalar, Value};

/// A memory that always produces the same instruction and data.
///
/// All operations will always succeed regardless of address or alignment.
///
/// The first member is the encoding of the instruction that the memory
/// should return from every `read_instr` while the second is what
/// `read_data` will populate all read bytes with.
///
/// Read/write metrics will be reported correctly, except alignment which will
/// be ignored.
///
///
/// Meant for testing.
#[derive(Debug, Clone, Copy)]
pub struct RepeatingMem(pub u16, pub u8);
impl Memory for RepeatingMem
{
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
		_: usize,
		from: &Value,
		tracker: &mut impl MetricTracker,
	) -> Result<(), (MemError, usize)>
	{
		tracker.add_stat(Metric::DataBytesWritten, from.size());
		Ok(())
	}
}
