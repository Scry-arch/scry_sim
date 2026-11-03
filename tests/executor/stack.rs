use crate::{
	executor::{load::idx_address, operand_queue_list_count, test_execution_step},
	misc::{add_to_queue_idx, advance_queue, regress_queue, OneShotTracker, RepeatingMem},
};
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_isa::{Bits, Instruction};
use scry_sim::{
	arbitrary::{LimitedOps, NoCF},
	BlockedMemory, CallFrameState, Event, EventTracker, ExecState, IdEvent, OperandEvent, Scalar,
	StackEvent, Value, ValueType,
};
use std::iter::once;

fn default_expected_state(state: &ExecState, instr: &Instruction)
	-> (ExecState, RepeatingMem<true>)
{
	let test_mem = RepeatingMem::<true>(instr.encode(), 0);

	let mut expected_state: ExecState = state.clone();
	expected_state.address += 2;
	expected_state.frame.op_queue = advance_queue(expected_state.frame.op_queue);
	(expected_state, test_mem)
}

/// Returns the events of stack reserve/free
fn stack_change_events(
	addr: usize,
	instr: Instruction,
	expected_frame: &CallFrameState,
	split_change: bool,
	top_change: bool,
) -> OneShotTracker
{
	let split_event = Event::Stack(
		StackEvent::Split,
		expected_frame.stack.block.address + expected_frame.stack.private_size,
	);
	let top_event = Event::Stack(
		StackEvent::Top,
		expected_frame.stack.block.address + expected_frame.stack.block.size,
	);

	let mut evs: OneShotTracker = [Event::Instruction(addr, instr)].into();

	if top_change
	{
		evs.event(top_event);
	}

	if split_change
	{
		evs.event(split_event);
	}

	evs
}

/// Tests can reserve additional shared stack frame
#[quickcheck]
fn static_reserve_shared(
	NoCF(LimitedOps(mut state)): NoCF<LimitedOps<ExecState, 0, 0>>,
	pow2: Bits<4, false>,
) -> TestResult
{
	let reserve_amount = 2usize.pow(pow2.value as u32);
	// Ensure stack buffer has some free bytes
	if state.stack_buffer < reserve_amount
	{
		state.stack_buffer += reserve_amount;
	}

	let instr = Instruction::StackRes(true, pow2, false);
	let (mut expected_state, test_mem) = default_expected_state(&state, &instr);
	expected_state.stack_buffer -= reserve_amount;
	expected_state.frame.stack.block.size += reserve_amount;

	test_execution_step(
		&state.clone(),
		test_mem,
		&expected_state,
		&stack_change_events(state.address, instr, &expected_state.frame, false, true),
	)
}

/// Tests can reserve private stack frame from the shared
#[quickcheck]
fn static_reserve_private(
	NoCF(LimitedOps(mut state)): NoCF<LimitedOps<ExecState, 0, 0>>,
	pow2: Bits<4, false>,
) -> TestResult
{
	let reserve_amount = 2usize.pow(pow2.value as u32);
	// Ensure stack buffer has some free bytes
	if state.stack_buffer < reserve_amount
	{
		state.stack_buffer += reserve_amount;
	}

	let instr = Instruction::StackRes(true, pow2, true);
	let (mut expected_state, test_mem) = default_expected_state(&state, &instr);
	let total_free = state.frame.stack.block.size - state.frame.stack.private_size;
	let total_res = if total_free < reserve_amount
	{
		reserve_amount - total_free
	}
	else
	{
		0
	};
	expected_state.stack_buffer -= total_res;
	expected_state.frame.stack.block.size += total_res;
	expected_state.frame.stack.private_size += reserve_amount;

	test_execution_step(
		&state.clone(),
		test_mem,
		&expected_state,
		&stack_change_events(
			state.address,
			instr,
			&expected_state.frame,
			true,
			total_res > 0,
		),
	)
}

/// Tests can free shared stack frame
#[quickcheck]
fn static_free_shared(
	NoCF(LimitedOps(mut state)): NoCF<LimitedOps<ExecState, 0, 0>>,
	pow2: Bits<4, false>,
) -> TestResult
{
	let free_amount = 2usize.pow(pow2.value as u32);
	// Ensure stack has size to free
	if state.frame.stack.block.size < free_amount
	{
		state.frame.stack.block.size += free_amount;
	}
	// Ensure stack buffer has room for the freed bytes
	if state.stack_buffer.checked_add(free_amount).is_none()
	{
		state.stack_buffer -= free_amount;
	}

	let instr = Instruction::StackRes(false, pow2, false);
	let (mut expected_state, test_mem) = default_expected_state(&state, &instr);
	expected_state.stack_buffer += free_amount;
	expected_state.frame.stack.block.size -= free_amount;
	let private_free =
		free_amount.saturating_sub(state.frame.stack.block.size - state.frame.stack.private_size);
	expected_state.frame.stack.private_size -= private_free;

	test_execution_step(
		&state.clone(),
		test_mem,
		&expected_state,
		&stack_change_events(
			state.address,
			instr,
			&expected_state.frame,
			private_free > 0,
			true,
		),
	)
}

/// Tests can free private stack frame
#[quickcheck]
fn static_free_private(
	NoCF(LimitedOps(mut state)): NoCF<LimitedOps<ExecState, 0, 0>>,
	pow2: Bits<4, false>,
) -> TestResult
{
	let free_amount = 2usize.pow(pow2.value as u32);
	// Ensure base stack has size to free
	if state.frame.stack.private_size < free_amount
	{
		state.frame.stack.block.size += free_amount;
		state.frame.stack.private_size += free_amount;
	}

	let instr = Instruction::StackRes(false, pow2, true);
	let (mut expected_state, test_mem) = default_expected_state(&state, &instr);
	expected_state.frame.stack.private_size -= free_amount;

	test_execution_step(
		&state.clone(),
		test_mem,
		&expected_state,
		&stack_change_events(state.address, instr, &expected_state.frame, true, false),
	)
}

/// Test getting a stack object's address using only index
#[quickcheck]
fn stack_address_indexed_static(
	NoCF(state): NoCF<ExecState>,
	size: Bits<2, false>,
	idx: Bits<5, false>,
) -> TestResult
{
	let effective_addr = idx_address(
		state.frame.stack.get_base_address(),
		2usize.pow(size.value as u32),
		idx.value as usize,
	);
	let result_type = ValueType::Uint(state.addr_space);
	let result_addr = Value::singleton_typed(
		result_type,
		Scalar::from_sized(effective_addr, result_type.scale()),
	);

	let mut test_state = state.clone();
	test_state.frame.op_queue = regress_queue(test_state.frame.op_queue);

	let mut expected_state: ExecState = state.clone();
	expected_state.address += 2;
	add_to_queue_idx(&mut expected_state, 0, once(result_addr));

	let instruction = Instruction::StackAddr(size, idx);
	let test_mem = BlockedMemory::new(
		instruction.encode().to_le_bytes().into_iter(),
		state.address,
	);

	let mut expected_events: OneShotTracker = [
		Event::Instruction(test_state.address, instruction),
		Event::Id(
			0,
			IdEvent::Operand(OperandEvent::Produce(result_type.into())),
		),
	]
	.into();

	if operand_queue_list_count(&state.frame.op_queue, 1) == state.op_list_depth()
	{
		expected_events.event(Event::Id(0, IdEvent::Operand(OperandEvent::Discard)));
	}
	test_execution_step(&test_state, test_mem, &expected_state, &expected_events)
}
