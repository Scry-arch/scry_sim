use crate::{executor::test_simple_instruction, misc::regress_queue};
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_isa::{Bits, Instruction};
use scry_sim::{
	arbitrary::{ArbValue, NoCF},
	ExecState, Metric, OperandList, OperandState, Value,
};

#[quickcheck]
fn pick_2_inputs(
	NoCF(state): NoCF<ExecState>,
	condition: ArbValue<false, false>,
	in1: Value,
	in2: Value,
	target: Bits<5, false>,
) -> TestResult
{
	let mut test_state = state.clone();
	test_state.frame.op_queue = regress_queue(test_state.frame.op_queue);
	test_state.frame.op_queue.insert(
		0,
		OperandList::new(
			OperandState::Ready(condition.0.clone()),
			vec![
				OperandState::Ready(in1.clone()),
				OperandState::Ready(in2.clone()),
			],
		),
	);

	test_simple_instruction(
		NoCF(test_state),
		Instruction::Pick(target),
		|old_op_queue| {
			let mut new_op_q = old_op_queue.clone();

			new_op_q.remove(&0); // Remove the operand we added above

			let choose_first = condition
				.0
				.get_first()
				.bytes()
				.unwrap()
				.iter()
				.all(|b| *b == 0);

			let chosen = if choose_first { in1 } else { in2 };

			let target_idx = (target.value + 1) as usize;
			if let Some(list) = new_op_q.get_mut(&target_idx)
			{
				list.rest.push(OperandState::Ready(chosen));
			}
			else
			{
				new_op_q.insert(
					target_idx,
					OperandList::new(OperandState::Ready(chosen), vec![]),
				);
			}

			new_op_q
		},
		|_| {
			[
				(Metric::ReorderedOperands, 1),
				(Metric::ConsumedOperands, 1),
				(Metric::ConsumedBytes, condition.0.size()),
			]
			.into()
		},
	)
}
