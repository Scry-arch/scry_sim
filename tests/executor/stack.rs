use crate::{
	executor::{load::idx_address, test_execution_step},
	misc::{advance_queue, regress_queue, RepeatingMem},
};
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_isa::{Bits, Instruction};
use scry_sim::{
	arbitrary::{LimitedOps, NoCF},
	BlockedMemory, ExecState, Metric, OperandList, Scalar, Value, ValueType,
};

fn default_expected_state(
	state: &ExecState,
	res: bool,
	pow2_amount: Bits<4, false>,
	base: bool,
) -> (ExecState, RepeatingMem<true>)
{
	let instr = Instruction::StackRes(res, pow2_amount, base);
	let test_mem = RepeatingMem::<true>(instr.encode(), 0);

	let mut expected_state: ExecState = state.clone();
	expected_state.address += 2;
	expected_state.frame.op_queue = advance_queue(expected_state.frame.op_queue);
	(expected_state, test_mem)
}

/// Tests can reserve additional total stack frame
#[quickcheck]
fn static_reserve_total(
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

	let (mut expected_state, test_mem) = default_expected_state(&state, true, pow2, false);
	expected_state.stack_buffer -= reserve_amount;
	expected_state.frame.stack.block.size += reserve_amount;

	test_execution_step(
		&state.clone(),
		test_mem,
		&expected_state,
		&[
			(Metric::InstructionReads, 1),
			(Metric::StackReserveTotal, 1),
			(Metric::StackReserveTotalBytes, reserve_amount),
		]
		.into(),
	)
}

/// Tests can reserve base stack frame all from the total
#[quickcheck]
fn static_reserve_base(
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

	let (mut expected_state, test_mem) = default_expected_state(&state, true, pow2, true);
	let total_free = state.frame.stack.block.size - state.frame.stack.base_size;
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
	expected_state.frame.stack.base_size += reserve_amount;

	test_execution_step(
		&state.clone(),
		test_mem,
		&expected_state,
		&[
			(Metric::InstructionReads, 1),
			(Metric::StackReserveBase, 1),
			(Metric::StackReserveTotalBytes, total_res),
			(Metric::StackReserveBaseBytes, reserve_amount),
		]
		.into(),
	)
}

/// Tests can free total stack frame
#[quickcheck]
fn static_free_total(
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

	let (mut expected_state, test_mem) = default_expected_state(&state, false, pow2, false);
	expected_state.stack_buffer += free_amount;
	expected_state.frame.stack.block.size -= free_amount;
	let base_free =
		free_amount.saturating_sub(state.frame.stack.block.size - state.frame.stack.base_size);
	expected_state.frame.stack.base_size -= base_free;

	test_execution_step(
		&state.clone(),
		test_mem,
		&expected_state,
		&[
			(Metric::InstructionReads, 1),
			(Metric::StackFreeTotal, 1),
			(Metric::StackFreeTotalBytes, free_amount),
			(Metric::StackFreeBaseBytes, base_free),
		]
		.into(),
	)
}

/// Tests can free base stack frame
#[quickcheck]
fn static_free_base(
	NoCF(LimitedOps(mut state)): NoCF<LimitedOps<ExecState, 0, 0>>,
	pow2: Bits<4, false>,
) -> TestResult
{
	let free_amount = 2usize.pow(pow2.value as u32);
	// Ensure base stack has size to free
	if state.frame.stack.base_size < free_amount
	{
		state.frame.stack.block.size += free_amount;
		state.frame.stack.base_size += free_amount;
	}

	let (mut expected_state, test_mem) = default_expected_state(&state, false, pow2, true);
	expected_state.frame.stack.base_size -= free_amount;

	test_execution_step(
		&state.clone(),
		test_mem,
		&expected_state,
		&[
			(Metric::InstructionReads, 1),
			(Metric::StackFreeBase, 1),
			(Metric::StackFreeBaseBytes, free_amount),
		]
		.into(),
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
		state.frame.stack.get_base_addres(),
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
	if let Some(list) = expected_state.frame.op_queue.get_mut(&0)
	{
		list.push(result_addr);
	}
	else
	{
		expected_state
			.frame
			.op_queue
			.insert(0, OperandList::new(result_addr, Vec::new()));
	}

	let instruction = Instruction::StackAddr(size, idx);
	let test_mem = BlockedMemory::new(
		instruction.encode().to_le_bytes().into_iter(),
		state.address,
	);

	test_execution_step(
		&test_state,
		test_mem,
		&expected_state,
		&[
			(Metric::InstructionReads, 1),
			(Metric::QueuedValues, 1),
			(Metric::QueuedValueBytes, state.pointer_size() as usize),
		]
		.into(),
	)
}
