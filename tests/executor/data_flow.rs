use crate::{
	executor::test_simple_instruction,
	misc::{add_list_to_queue_idx, add_value_to_queue_idx},
};
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_isa::{Bits, Instruction};
use scry_sim::{arbitrary::NoCF, ExecState, Metric, OperandList};
use std::cmp::min;

/// Tests the "duplicate" instruction
#[quickcheck]
fn duplicate(
	state: NoCF<ExecState>,
	target1: Bits<5, false>,
	target2: Bits<5, false>,
	dup_next: bool,
) -> TestResult
{
	let op_list_depth = state.0.op_list_depth();
	test_simple_instruction(
		state,
		Instruction::Duplicate(dup_next, target1, target2),
		|old_op_queue| {
			let mut new_op_q = old_op_queue.clone();

			if let Some(old_ops) = new_op_q.remove(&0)
			{
				// Push a copy of old_ready_ops to the list of the given index
				let mut push_idx = |idx| {
					add_list_to_queue_idx(
						&mut new_op_q,
						op_list_depth,
						idx,
						old_ops.clone().into_iter(),
					);
				};

				if dup_next
				{
					push_idx(1);
				}
				push_idx(target1.value as usize + 1);
				push_idx(target2.value as usize + 1);
			}
			new_op_q
		},
		|old_op_queue| {
			let (old_op_ready, old_op_ready_bytes) =
				old_op_queue.get(&0).map_or((0, 0), |op_list| {
					op_list.iter().fold((0, 0), |(ready, read_bytes), v| {
						(ready + 1, v.value_type().scale() + read_bytes)
					})
				});
			let double_if_dup_next = |val| if dup_next { val * 2 } else { val };
			[
				(Metric::QueuedValues, double_if_dup_next(old_op_ready)),
				(
					Metric::QueuedValueBytes,
					double_if_dup_next(old_op_ready_bytes),
				),
				(Metric::ReorderedOperands, old_op_ready),
			]
			.into()
		},
	)
}

/// Tests the "echo" instruction
#[quickcheck]
fn echo(
	state: NoCF<ExecState>,
	target1: Bits<5, false>,
	target2: Bits<5, false>,
	rest_next: bool,
) -> TestResult
{
	let op_list_depth = state.0.op_list_depth();
	test_simple_instruction(
		state,
		Instruction::Echo(rest_next, target1, target2),
		|old_op_queue| {
			let mut new_op_q = old_op_queue.clone();

			if let Some(old_ops) = new_op_q.remove(&0)
			{
				let mut iter = old_ops.into_iter();

				if let Some(first) = iter.next()
				{
					// First add the second operand, this ensures it comes before the first
					// if they are going to the same list
					if let Some(second) = iter.next()
					{
						let tar2_idx = target2.value as usize + 1;
						add_value_to_queue_idx(&mut new_op_q, op_list_depth, tar2_idx, second);
					}
					let tar1_idx = target1.value as usize + 1;
					add_value_to_queue_idx(&mut new_op_q, op_list_depth, tar1_idx, first);
					// handle the rest if needed
					if rest_next
					{
						add_list_to_queue_idx(&mut new_op_q, op_list_depth, 1, iter);
					}
				}
			}

			new_op_q
		},
		|old_op_queue| {
			let reordered = old_op_queue.get(&0).map_or(0, |list| {
				if rest_next
				{
					list.len()
				}
				else
				{
					min(2, list.len())
				}
			});
			[(Metric::ReorderedOperands, reordered)].into()
		},
	)
}

/// Tests the long "echo" instruction
#[quickcheck]
fn echo_long(state: NoCF<ExecState>, target: Bits<10, false>) -> TestResult
{
	let op_list_depth = state.0.op_list_depth();
	test_simple_instruction(
		state,
		Instruction::EchoLong(target),
		|old_op_queue| {
			let mut new_op_q = old_op_queue.clone();

			if let Some(old_ops) = new_op_q.remove(&0)
			{
				let iter = old_ops.into_iter();

				let tar_idx = target.value as usize + 1;
				add_list_to_queue_idx(&mut new_op_q, op_list_depth, tar_idx, iter);
			}

			new_op_q
		},
		|old_op_queue| {
			let reordered = old_op_queue.get(&0).map_or(0, OperandList::len);
			[(Metric::ReorderedOperands, reordered)].into()
		},
	)
}

/// Test the "NoOp" instruction
#[quickcheck]
fn noop(state: NoCF<ExecState>) -> TestResult
{
	test_simple_instruction(
		state,
		Instruction::NoOp,
		|old_op_queue| {
			let mut new_op_q = old_op_queue.clone();
			// discard ready list
			let _ = new_op_q.remove(&0);

			new_op_q
		},
		|_| [].into(),
	)
}
