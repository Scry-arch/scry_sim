use crate::{
	executor::test_execution_step,
	misc::{advance_queue, RepeatingMem},
};
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_isa::{Bits, Instruction};
use scry_sim::{
	arbitrary::{LimitedOps, NoCF},
	ExecState, Metric,
};

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

	let instr = Instruction::StackRes(true, false, pow2);
	let test_mem = RepeatingMem::<true>(instr.encode(), 0);

	let mut expected_state: ExecState = state.clone();
	expected_state.address += 2;
	expected_state.frame.op_queue = advance_queue(expected_state.frame.op_queue);
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

	let instr = Instruction::StackRes(false, false, pow2);
	let test_mem = RepeatingMem::<true>(instr.encode(), 0);

	let mut expected_state: ExecState = state.clone();
	expected_state.address += 2;
	expected_state.frame.op_queue = advance_queue(expected_state.frame.op_queue);
	expected_state.stack_buffer += free_amount;
	expected_state.frame.stack.block.size -= free_amount;

	test_execution_step(
		&state.clone(),
		test_mem,
		&expected_state,
		&[
			(Metric::InstructionReads, 1),
			(Metric::StackFreeTotal, 1),
			(Metric::StackFreeTotalBytes, free_amount),
		]
		.into(),
	)
}
