use byteorder::{ByteOrder, LittleEndian};
use duplicate::duplicate_item;
use quickcheck_macros::quickcheck;
use scry_sim::{MemError, Memory, Metric, MetricTracker, OperandQueue, Scalar, Value, ValueType};
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
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
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

		tracker.add_stat(Metric::DataReadBytes, result.size());

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
			AllowWrite(self).write(addr, from, tracker)
		}
		else
		{
			Err((MemError::InvalidAddr, addr))
		}
	}
}

/// Wrapper around a memory that accept all writes and allows them to succeed
/// regardless og address.
#[derive(Debug)]
pub struct AllowWrite<'a, M: Memory>(pub &'a mut M);
impl<'a, M: Memory> Memory for AllowWrite<'a, M>
{
	delegate::delegate! {
		to self.0{
			fn read_raw(&mut self, addr: usize) -> Option<u8>;
			fn read_data(
				&mut self,
				addr: usize,
				into: &mut Value,
				len: usize,
				tracker: &mut impl MetricTracker,
			) -> Result<(), (MemError, usize)>;
			fn read_instr(
				&mut self,
				addr: usize,
				tracker: &mut impl MetricTracker,
			) -> Result<[u8; 2], MemError>;
		}
	}

	fn write(
		&mut self,
		_: usize,
		from: &Value,
		tracker: &mut impl MetricTracker,
	) -> Result<(), (MemError, usize)>
	{
		tracker.add_stat(Metric::DataWriteBytes, from.size());
		Ok(())
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

/// Interprets the given scalar as a usize.
pub fn as_usize(operand: &Scalar) -> usize
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

/// Interprets the given scalar as a isize.
pub fn as_isize(operand: &Scalar) -> isize
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

	// If negative, must sign extend
	if *operand.bytes().unwrap().last().unwrap() > (i8::MAX as u8)
		&& size_of::<usize>() > operand.bytes().unwrap().len()
	{
		let mut extending_bits = usize::MAX;
		extending_bits = extending_bits
			.overflowing_shl(8 * operand.bytes().unwrap().len() as u32)
			.0;
		result += extending_bits;
	}

	// Reinterpret as isize
	isize::from_le_bytes(result.to_le_bytes())
}

/// Get the address represented by the given value.
///
/// The value is assumed to be an unsigned integer.
pub fn get_absolute_address(addr: &Scalar) -> usize
{
	as_usize(addr)
}

/// Get the absolute address from the current address and the given offset
/// value.
///
/// The value is assumed to be a signed integer
pub fn get_relative_address(current_address: usize, offset: &Scalar) -> Option<usize>
{
	let offset_value = as_isize(offset);
	if offset_value < 0
	{
		current_address.checked_sub(offset_value.abs_diff(0))
	}
	else
	{
		current_address.checked_add(offset_value.abs_diff(0))
	}
}

/// Get the absolute address from the given indexed address.
///
/// The base address may me signed or unsiged while the index is assumed to be
/// unsigned. In case of relative base address, the given current address is
/// used.
pub fn get_indexed_address(
	current_address: usize,
	base_addr: &Value,
	index: &Scalar,
	index_scale: usize,
) -> Option<usize>
{
	let checked_absolute_addr = if let ValueType::Int(_) = base_addr.value_type()
	{
		get_relative_address(current_address, base_addr.get_first())
	}
	else
	{
		Some(get_absolute_address(base_addr.get_first()))
	};
	let offset = as_usize(&index).checked_mul(index_scale);

	checked_absolute_addr.and_then(|addr| offset.and_then(|offset| addr.checked_add(offset)))
}

/// Test as_usize on various type lengths
#[duplicate_item(
	name 					typ;
	[test_as_usize_u8] 		[u8];
	[test_as_usize_u16] 	[u16];
	[test_as_usize_u32]  	[u32];
	[test_as_usize_usize]	[usize];
)]
#[quickcheck]
fn name(operand: typ) -> bool
{
	let mut scal = Scalar::Nan;
	scal.set_val(&operand.to_le_bytes());
	let result = as_usize(&scal);
	result == (operand as usize)
}

/// Test as_isize on various type lengths
#[duplicate_item(
	name 					typ;
	[test_as_isize_i8] 		[i8];
	[test_as_isize_i16] 	[i16];
	[test_as_isize_i32]  	[i32];
	[test_as_isize_isize]	[isize];
)]
#[quickcheck]
fn name(operand: typ) -> bool
{
	let mut scal = Scalar::Nan;
	scal.set_val(&operand.to_le_bytes());
	let result = as_isize(&scal);
	result == (operand as isize)
}
