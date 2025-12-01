use crate::{
	executor::test_execution_step,
	misc::{regress_queue, RepeatingMem},
};
use duplicate::duplicate_item;
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_isa::{Bits, Instruction};
use scry_sim::{
	arbitrary::{ArbValue, NoCF},
	ExecState, Metric, OperandList, Scalar, Value, ValueType,
};

/// Tests the cast instruction
#[duplicate_item(
	name			to_typ	typ_encoding;
	[cast_to_u8]	[u8]	[0b0000];
	[cast_to_u16]	[u16]	[0b0010];
	[cast_to_u32]	[u32]	[0b0100];
	[cast_to_u64]	[u64]	[0b0110];
	[cast_to_i8]	[i8]	[0b0001];
	[cast_to_i16]	[i16]	[0b0011];
	[cast_to_i32]	[i32]	[0b0101];
	[cast_to_i64]	[i64]	[0b0111];
)]
#[quickcheck]
fn name(
	NoCF(mut state): NoCF<ExecState>,
	ArbValue(to_cast): ArbValue<false, false>,
	out_off: Bits<5, false>,
) -> TestResult
{
	let target_type = ValueType::new::<to_typ>();
	let to_cast_bytes = to_cast.first.bytes().unwrap();

	// Calculate result bytes
	let mut target_bytes = Vec::new();
	if to_cast.typ.scale() < target_type.scale()
	{
		for i in 0..to_cast.typ.scale()
		{
			target_bytes.push(to_cast_bytes[i]);
		}
		// Need extension
		let extender = if to_cast.typ.is_signed_integer()
			&& to_cast_bytes.last().unwrap() >= &0b1000_0000u8
		{
			u8::MAX
		}
		else
		{
			0
		};
		for _ in to_cast.typ.scale()..target_type.scale()
		{
			target_bytes.push(extender);
		}
	}
	else
	{
		for i in 0..target_type.scale()
		{
			target_bytes.push(to_cast_bytes[i]);
		}
	}
	assert_eq!(target_bytes.len(), target_type.scale());

	let target_value = Value::singleton::<to_typ>(Scalar::Val(target_bytes.into_boxed_slice()));

	let mut expected_state = state.clone();
	expected_state.foli = target_value.clone();
	if let Some(list) = expected_state
		.frame
		.op_queue
		.get_mut(&(out_off.value as usize))
	{
		list.rest.push(target_value)
	}
	else
	{
		expected_state.frame.op_queue.insert(
			out_off.value as usize,
			OperandList::new(target_value, vec![]),
		);
	}
	expected_state.address += 2;

	state.frame.op_queue = regress_queue(state.frame.op_queue);
	state
		.frame
		.op_queue
		.insert(0, OperandList::new(to_cast.clone(), vec![]));

	test_execution_step(
		&state,
		RepeatingMem::<true>(
			Instruction::Cast(typ_encoding.try_into().unwrap(), out_off).encode(),
			0,
		),
		&expected_state,
		&([
			(Metric::ConsumedBytes, to_cast_bytes.len()),
			(Metric::ConsumedOperands, 1),
			(Metric::QueuedValueBytes, size_of::<to_typ>()),
			(Metric::QueuedValues, 1),
			(Metric::InstructionReads, 1),
		]
		.into()),
	)
}
