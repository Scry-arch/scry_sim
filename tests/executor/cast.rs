use crate::{
	executor::test_execution_step,
	misc::{add_to_queue_idx, regress_queue, OneShotTracker, RepeatingMem},
};
use duplicate::duplicate_item;
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_isa::{Bits, Instruction};
use scry_sim::{
	arbitrary::{ArbValue, NoCF},
	Event, EventTracker, ExecState, IdEvent, OperandEvent, OperandList, Scalar, Value, ValueType,
};
use std::iter::once;

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

	let mut expected_events = OneShotTracker::new();
	let target_op = target_value.into_operand(&mut expected_events);
	expected_events.event(Event::Id(
		target_op.id,
		IdEvent::Operand(OperandEvent::Produce),
	));

	let mut expected_state = state.clone();
	if 0 == add_to_queue_idx(
		&mut expected_state,
		out_off.value as usize,
		once(target_op.get_value().clone()),
	)
	{
		expected_events.event(Event::Id(
			target_op.id,
			IdEvent::Operand(OperandEvent::Discard),
		));
	}

	expected_state.address += 2;

	state.frame.op_queue = regress_queue(state.frame.op_queue);
	state
		.frame
		.op_queue
		.insert(0, OperandList::new(to_cast.clone(), vec![]));
	let to_cast_id = expected_events.new_operand_id();
	expected_events.event(Event::Id(
		to_cast_id,
		IdEvent::Operand(OperandEvent::Consume),
	));

	let instr = Instruction::Cast(typ_encoding.try_into().unwrap(), out_off);
	expected_events.event(Event::Instruction(state.address, instr.clone()));
	test_execution_step(
		&state,
		RepeatingMem::<true>(instr.encode(), 0),
		&expected_state,
		&expected_events,
	)
}
