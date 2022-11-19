use crate::executor::test_simple_instruction;
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_isa::{Bits, Instruction};
use scry_sim::{arbitrary::NoCF, ExecState, Metric, OperandList, OperandState};
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
	test_simple_instruction(
		state,
		Instruction::Duplicate(dup_next, target1, target2),
		|old_op_queue| {
			let mut new_op_q = old_op_queue.clone();

			if let Some(old_ops) = new_op_q.remove(&0)
			{
				// Push a copy of old_ready_ops to the list of the given index
				let mut push_idx = |idx| {
					if let Some(ops) = new_op_q.get_mut(&idx)
					{
						ops.extend(old_ops.clone().into_iter());
					}
					else
					{
						new_op_q.insert(idx, old_ops.clone());
					}
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
			let (old_op_ready, old_op_ready_bytes, old_op_reads) =
				old_op_queue.get(&0).map_or((0, 0, 0), |op_list| {
					op_list
						.iter()
						.fold((0, 0, 0), |(ready, read_bytes, reads), op| {
							match op
							{
								OperandState::MustRead(_) => (ready, read_bytes, reads + 1),
								OperandState::Ready(v) =>
								{
									(ready + 1, v.value_type().scale() + read_bytes, reads)
								},
							}
						})
				});
			let double_if_dup_next = |val| if dup_next { val * 2 } else { val };
			[
				(Metric::QueuedValues, double_if_dup_next(old_op_ready)),
				(
					Metric::QueuedValueBytes,
					double_if_dup_next(old_op_ready_bytes),
				),
				(Metric::QueuedReads, double_if_dup_next(old_op_reads)),
				(Metric::ReorderedOperands, old_op_ready + old_op_reads),
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
						if let Some(list) = new_op_q.get_mut(&tar2_idx)
						{
							list.push(second);
						}
						else
						{
							new_op_q.insert(tar2_idx, OperandList::new(second, vec![]));
						}
					}
					let tar1_idx = target1.value as usize + 1;
					if let Some(list) = new_op_q.get_mut(&tar1_idx)
					{
						list.push(first);
					}
					else
					{
						new_op_q.insert(tar1_idx, OperandList::new(first, vec![]));
					}
					// handle the rest if needed
					if rest_next
					{
						if let Some(list) = new_op_q.get_mut(&1)
						{
							list.extend(iter);
						}
						else if let Some(first_rest) = iter.next()
						{
							new_op_q.insert(1, OperandList::new(first_rest, iter.collect()));
						}
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

/// Tests the "echo" instruction
#[quickcheck]
fn echo_long(state: NoCF<ExecState>, target: Bits<10, false>) -> TestResult
{
	test_simple_instruction(
		state,
		Instruction::EchoLong(target),
		|old_op_queue| {
			let mut new_op_q = old_op_queue.clone();

			if let Some(old_ops) = new_op_q.remove(&0)
			{
				let mut iter = old_ops.into_iter();

				let tar_idx = target.value as usize + 1;
				if let Some(list) = new_op_q.get_mut(&tar_idx)
				{
					list.extend(iter);
				}
				else if let Some(first) = iter.next()
				{
					new_op_q.insert(tar_idx, OperandList::new(first, iter.collect()));
				}
			}

			new_op_q
		},
		|old_op_queue| {
			let reordered = old_op_queue.get(&0).map_or(0, OperandList::len);
			[(Metric::ReorderedOperands, reordered)].into()
		},
	)
}

/// Test the "Capture" instruction
#[quickcheck]
fn capture(state: NoCF<ExecState>, cap: Bits<5, false>, target: Bits<5, false>) -> TestResult
{
	let cap_idx = cap.value as usize + 1;
	test_simple_instruction(
		state,
		Instruction::Capture(cap, target),
		|old_op_queue| {
			let mut new_op_q = old_op_queue.clone();

			if let Some(old_ops) = new_op_q.remove(&cap_idx)
			{
				let mut iter = old_ops.into_iter();

				let tar_idx = target.value as usize + 1;
				if let Some(list) = new_op_q.get_mut(&tar_idx)
				{
					list.extend(iter);
				}
				else if let Some(first) = iter.next()
				{
					new_op_q.insert(tar_idx, OperandList::new(first, iter.collect()));
				}
			}
			// discard ready list
			let _ = new_op_q.remove(&0);

			new_op_q
		},
		|old_op_queue| {
			let reordered = old_op_queue.get(&cap_idx).map_or(0, OperandList::len);
			[(
				Metric::ReorderedOperands,
				if cap.value != target.value
				{
					reordered
				}
				else
				{
					0
				},
			)]
			.into()
		},
	)
}
