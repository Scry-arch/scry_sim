use crate::{
	executor::{operand_queue_list_count, test_simple_instruction},
	misc::{add_value_to_queue_idx, regress_queue, OneShotTracker},
};
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_isa::{Bits, Instruction};
use scry_sim::{
	arbitrary::{ArbValue, NoCF},
	Event, EventTracker, ExecState, IdEvent, OperandEvent, OperandList, Value,
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

	let target_idx = (target.value + 1) as usize;
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

			add_value_to_queue_idx(&mut new_op_q, op_list_depth, target_idx, chosen);

			new_op_q
		},
		|old_op_queue| {
			let mut evs: OneShotTracker = [
				Event::Id(0, IdEvent::Operand(OperandEvent::Consume)),
				Event::Id(1, IdEvent::Operand(OperandEvent::Forward)),
				Event::Id(2, IdEvent::Operand(OperandEvent::Discard)),
			]
			.into();

			if operand_queue_list_count(old_op_queue, target_idx) == op_list_depth
			{
				evs.event(Event::Id(1, IdEvent::Operand(OperandEvent::Forward)));
			}
			evs
		},
	)
}
