use crate::{
	executor::test_simple_instruction,
	misc::{add_value_to_queue_idx, regress_queue},
};
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_isa::{Bits, Instruction};
use scry_sim::{
	arbitrary::{ArbValue, NoCF},
	ExecState, Metric, OperandList, Value,
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
	let op_list_depth = state.op_list_depth();
	let mut test_state = state.clone();
	test_state.frame.op_queue = regress_queue(test_state.frame.op_queue);
	test_state.frame.op_queue.insert(
		0,
		OperandList::new(condition.0.clone(), vec![in1.clone(), in2.clone()]),
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
			add_value_to_queue_idx(&mut new_op_q, op_list_depth, target_idx, chosen);

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
