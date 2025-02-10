use crate::{
	executor::test_execution_step,
	misc::{advance_queue, RepeatingMem},
};
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_isa::Instruction;
use scry_sim::{
	arbitrary::{LimitedOps, NoCF},
	Block, ExecState, Executor, Metric,
};
use std::convert::TryInto;

/// Tests can reserve additional total stack frame
#[quickcheck]
fn static_reserve_total(
	NoCF(LimitedOps(state)): NoCF<LimitedOps<ExecState, 0, 0>>,
	pow2: u8,
) -> TestResult
{
	let mut pow2 = pow2 as i32;
	// Ensure stack buffer has some free blocks
	if state.stack_buffer.is_empty()
	{
		return TestResult::discard();
	}

	// Determine the actual amount to reserve based on availability
	let mut reserve = pow2.try_into();
	while reserve.is_err() || 2usize.pow(pow2 as u32) > state.stack_buffer.last().unwrap().size
	{
		pow2 -= 1;
		reserve = pow2.try_into();
	}

	let instr = Instruction::StackRes(true, false, reserve.unwrap());
	let test_mem = RepeatingMem::<true>(instr.encode(), 0);

	let mut expected_state: ExecState = state.clone();
	expected_state.address += 2;
	expected_state.frame.op_queue = advance_queue(expected_state.frame.op_queue);

	// Because executor equality depends on the order of the stack buffer,
	// run the executer and extract the buffer
	let executed = Executor::from_state(&state, test_mem)
		.step(&mut ())
		.unwrap()
		.state();
	expected_state.stack_buffer = executed.stack_buffer;
	// To not be dependent on which exact address is reserved, take it from executer
	let reserve_size = 2usize.pow(pow2 as u32);
	expected_state.frame.stack.blocks.push(Block {
		address: executed.frame.stack.blocks.last().unwrap().address,
		size: reserve_size,
	});

	test_execution_step(
		&state.clone(),
		test_mem,
		&expected_state,
		&[
			(Metric::InstructionReads, 1),
			(Metric::StackReserveTotal, 1),
			(Metric::StackReserveTotalBytes, reserve_size),
		]
		.into(),
	)
}
