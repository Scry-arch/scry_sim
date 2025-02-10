use crate::{
	executor::test_execution_step,
	misc::{advance_queue, RepeatingMem},
};
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_isa::{Bits, Instruction};
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

/// Tests can free total stack frame
#[quickcheck]
fn static_free_total(
	NoCF(LimitedOps(state)): NoCF<LimitedOps<ExecState, 0, 0>>,
	address: usize,
	size_pow2: Bits<4, false>,
) -> TestResult
{
	let mut test_state = state.clone();
	let free_bytes = 2usize.pow(size_pow2.value as u32);
	let mut freed_block = Block {
		address,
		size: free_bytes,
	};
	// if the size is too large, shrink to fit
	while freed_block.validate().is_err()
	{
		freed_block.size /= 2;
		if freed_block.size == 0
		{
			return TestResult::discard();
		}
	}
	test_state.frame.stack.blocks.push(freed_block.clone());

	// If the block clashes with some other block in the frame, discard
	if test_state.validate().is_err()
	{
		return TestResult::discard();
	}

	let instr = Instruction::StackRes(false, false, size_pow2.clone());
	let test_mem = RepeatingMem::<true>(instr.encode(), 0);

	let mut expected_state: ExecState = state.clone();
	expected_state.address += 2;
	expected_state.frame.op_queue = advance_queue(expected_state.frame.op_queue);

	if expected_state
		.stack_buffer
		.last()
		.map(|b| b.address == (address + free_bytes))
		.unwrap_or(false)
	{
		let buf_top = expected_state.stack_buffer.last_mut().unwrap();
		*buf_top = Block {
			address,
			size: free_bytes + buf_top.size,
		};
	}
	else
	{
		expected_state.stack_buffer.push(freed_block)
	}

	test_execution_step(
		&test_state,
		test_mem,
		&expected_state,
		&[
			(Metric::InstructionReads, 1),
			(Metric::StackFreeTotal, 1),
			(Metric::StackFreeTotalBytes, free_bytes),
		]
		.into(),
	)
}
