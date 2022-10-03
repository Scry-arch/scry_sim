use crate::executor::test_simple_instruction;
use quickcheck::TestResult;
use scry_isa::{Bits, Instruction};
use scry_sim::{arbitrary::NoCF, ExecState, Metric, OperandState};

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
