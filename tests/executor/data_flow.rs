use crate::{
	executor::{add_discards, operand_events_for_list, test_simple_instruction},
	misc::{add_list_to_queue_idx, add_value_to_queue_idx, OneShotTracker},
};
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_isa::{Bits, Instruction};
use scry_sim::{arbitrary::NoCF, Event, EventTracker, ExecState, IdEvent, OperandEvent};
use std::collections::HashMap;

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
	let tar1_idx = target1.value as usize + 1;
	let tar2_idx = target2.value as usize + 1;

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
			let mut evs = OneShotTracker::new();
			let mut target_queues = HashMap::new();

			if let Some(list) = old_op_queue.get(&0)
			{
				let ops = list
					.iter()
					.map(|v| v.clone().into_operand(&mut evs))
					.enumerate()
					.collect::<Vec<_>>();

				let mut targets = vec![
					(tar1_idx, false, OperandEvent::Forward),
					(tar2_idx, true, OperandEvent::Produce),
				];
				if dup_next
				{
					targets.push((1, true, OperandEvent::Produce));
				}

				for (tar_idx, new_ids, ty) in targets
				{
					for (_, op) in ops.iter()
					{
						let op_id = if new_ids { evs.new_operand_id() } else { op.id };

						evs.event(Event::Id(op_id, IdEvent::Operand(ty)));
						target_queues
							.entry(tar_idx)
							.or_insert(Vec::new())
							.push(op_id);
					}
				}

				add_discards(&mut evs, &target_queues, old_op_queue, op_list_depth);
			}

			evs
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
	let tar1_idx = target1.value as usize + 1;
	let tar2_idx = target2.value as usize + 1;

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
					// First add the second operand, this ensures it comes
					// before the first if they are going to the same list
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
			let mut evs = OneShotTracker::new();
			let mut target_queues = HashMap::new();

			if let Some(list) = old_op_queue.get(&0)
			{
				let mut ops = list.iter().map(|v| v.clone().into_operand(&mut evs));

				let mut targets = vec![(tar1_idx, ops.next().unwrap())];
				ops.next().map(|op| targets.push((tar2_idx, op)));

				for op in ops.collect::<Vec<_>>()
				{
					if rest_next
					{
						targets.push((1, op));
					}
					else
					{
						evs.event(Event::Id(op.id, IdEvent::Operand(OperandEvent::Discard)));
					}
				}

				for (tar_idx, op) in targets
				{
					evs.event(Event::Id(op.id, IdEvent::Operand(OperandEvent::Forward)));
					target_queues
						.entry(tar_idx)
						.or_insert(Vec::new())
						.push(op.id);
				}

				add_discards(&mut evs, &target_queues, old_op_queue, op_list_depth);
			}

			evs
		},
	)
}

/// Tests the long "echo" instruction
#[quickcheck]
fn echo_long(state: NoCF<ExecState>, target: Bits<10, false>) -> TestResult
{
	let op_list_depth = state.0.op_list_depth();
	let tar_idx = target.value as usize + 1;
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
			let list_count = |idx| old_op_queue.get(&idx).map_or(0, |list| list.len());

			let mut evs = OneShotTracker::new();

			if let Some(list) = old_op_queue.get(&0)
			{
				for (idx, val) in list.iter().enumerate()
				{
					let op = val.clone().into_operand(&mut evs);
					evs.event(Event::Id(op.id, IdEvent::Operand(OperandEvent::Forward)));

					if list_count(tar_idx) + idx >= op_list_depth
					{
						evs.event(Event::Id(op.id, IdEvent::Operand(OperandEvent::Discard)));
					}
				}
			}

			evs
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
		|old_op_queue| operand_events_for_list(old_op_queue, 0, 0, &[OperandEvent::Discard]),
	)
}
