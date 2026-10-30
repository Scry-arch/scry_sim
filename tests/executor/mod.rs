use crate::misc::{
	add_list_to_queue_idx, add_to_queue_idx, add_value_to_queue_idx, advance_queue, OneShotTracker,
	RepeatingMem,
};
use quickcheck::{Arbitrary, Gen, TestResult};
use quickcheck_macros::quickcheck;
use scry_isa::{Alu2Variant, AluVariant, Bits, CallVariant, Instruction, Type};
use scry_sim::{
	arbitrary::{LimitedOps, NoCF, SimpleOps},
	Event, EventTracker, ExecError, ExecState, Executor, IdEvent, Memory, Metric, MetricReporter,
	MetricTracker, OperandEvent, OperandQueue, Scalar, TrackReport, Value, ValueType,
};
use std::{
	borrow::BorrowMut,
	collections::HashMap,
	fmt::Debug,
	iter::{once, repeat},
};

mod alu_instructions;
mod cast;
mod control_flow;
mod data_flow;
mod load;
mod pick;
mod stack;
mod store;

/// Returns the number of operands in the operand list of the given index in the
/// given queue.
fn operand_queue_list_count(queue: &OperandQueue, idx: usize) -> usize
{
	queue.get(&idx).map_or(0, |list| list.len())
}

/// Adds the given operand IDs and event types to the tracker.
/// If the number of events exceeds the given depth in the queue of the given
/// existing length, will add discard events for the overflowing IDs.
fn add_events_with_discards(
	evs: &mut OneShotTracker,
	ops: Vec<(usize, OperandEvent)>,
	queue_len: usize,
	queue_depth: usize,
)
{
	for (idx, (id, ty)) in ops.into_iter().enumerate()
	{
		evs.event(Event::Id(id, IdEvent::Operand(ty)));
		if queue_len + idx >= queue_depth
		{
			evs.event(Event::Id(id, IdEvent::Operand(OperandEvent::Discard)));
		}
	}
}

/// Adds discard events to the tracker for each target operand list that
/// overflows the given depth.
///
/// `target_queues` maps a target operand list index in the given queue to the
/// operand IDs that are going into the list. When the number of operands in a
/// list reaches the depth, subsequent operands get a discard event.
pub fn add_discards(
	evs: &mut OneShotTracker,
	target_queues: &HashMap<usize, Vec<usize>>,
	queue: &OperandQueue,
	queue_depth: usize,
)
{
	for (tar_idx, ops) in target_queues
	{
		for (idx, id) in ops.iter().enumerate()
		{
			if operand_queue_list_count(queue, *tar_idx) + idx >= queue_depth
			{
				evs.event(Event::Id(*id, IdEvent::Operand(OperandEvent::Discard)));
			}
		}
	}
}

/// Used to generate arbitrary instruction that are supported by the
/// executor.
///
/// Supported means that for any input combinations the instruction will
/// execute without panicking.
/// The result is allowed to be a simulation exception.
#[derive(Debug, Clone)]
struct SupportedInstruction(Instruction);
impl Arbitrary for SupportedInstruction
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		let mut instr: Instruction = Instruction::arbitrary(g);

		loop
		{
			use Instruction::*;
			match instr
			{
				Alu(AluVariant::Add, _)
				| Alu(AluVariant::Sub, _)
				| Alu2(Alu2Variant::Add, _, _)
				| Alu2(Alu2Variant::Sub, _, _)
				| Constant(..)
				| Call(CallVariant::Ret, _)
				| Duplicate(..)
				| Echo(..)
				| EchoLong(..)
				| NoOp => break,
				_ => instr = Instruction::arbitrary(g),
			}
		}
		Self(instr)
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		Box::new(self.0.shrink().map(|shrunk| Self(shrunk)))
	}
}

/// Checks that the expected metrics are equal to the actual ones.
/// If not, prints relevant error
pub fn test_metrics(
	expected: &TrackReport,
	actual: &TrackReport,
	expected_ev: &OneShotTracker,
	actual_ev: &OneShotTracker,
) -> TestResult
{
	if actual != expected
	{
		TestResult::error(format!(
			"Unexpected step metrics (actual != expected):\n{:?}\n !=\n {:?}",
			actual, expected
		))
	}
	else if actual_ev != expected_ev
	{
		TestResult::error(format!(
			"Unexpected step events (actual != expected):\n{:?}\n !=\n {:?}",
			actual_ev, expected_ev
		))
	}
	else
	{
		TestResult::passed()
	}
}

/// Performs 1 execution step with the given start state and memory.
/// Then checks that the given expected state is achieved with the expected
/// metrics reported.
pub fn test_execution_step<M: Memory + Debug>(
	start_state: &ExecState,
	start_memory: impl BorrowMut<M> + Debug,
	expected_state: &ExecState,
	expected_metrics: &TrackReport,
	expected_events: &OneShotTracker,
) -> TestResult
{
	let mut actual_metrics = TrackReport::new();
	let mut actual_events = OneShotTracker::new();
	match Executor::from_state(start_state, start_memory, &mut actual_events)
		.step(&mut actual_metrics, &mut actual_events)
	{
		Ok(exec) =>
		{
			check_expected(
				&exec.state(),
				expected_state,
				|| {
					test_metrics(
						expected_metrics,
						&actual_metrics,
						&expected_events,
						&actual_events,
					)
				},
				"end state",
			)
		},
		err => TestResult::error(format!("Test step failed: {:?}", err)),
	}
}

/// Checks that the actual value is equal to the given.
///
/// If not, returns a relevant error message.
/// If equal, runs the given closure and returns its result.
/// The message should describe what was unexpected.
pub fn check_expected<P: PartialEq + Debug>(
	actual: P,
	expected: P,
	then: impl Fn() -> TestResult,
	msg: &str,
) -> TestResult
{
	if actual != expected
	{
		TestResult::error(format!(
			"Unexpected {} (actual != expected):\n{:?}\n !=\n {:?}",
			msg, actual, expected
		))
	}
	else
	{
		then()
	}
}

/// Tests that an execution step from the given state/memory results in an
/// exception with the given expected metrics.
pub fn test_execution_step_exceptions<M: Memory + Debug>(
	start_state: &ExecState,
	start_memory: impl BorrowMut<M> + Debug,
	expected_metrics: &TrackReport,
	expected_events: &OneShotTracker,
) -> TestResult
{
	let mut actual_metrics = TrackReport::new();
	let mut actual_events = OneShotTracker::new();
	match Executor::from_state(start_state, start_memory, &mut actual_events)
		.step(&mut actual_metrics, &mut actual_events)
	{
		Err(ExecError::Exception(_)) =>
		{
			test_metrics(
				expected_metrics,
				&actual_metrics,
				expected_events,
				&actual_events,
			)
		},
		result =>
		{
			TestResult::error(format!(
				"Execution expected to end in an exception. Was: {:?}",
				result
			))
		},
	}
}

/// Tests a "simple" instruction, which must not:
///
/// * Issue control flow
/// * Consume operands (may discard or reorder)
/// * Only affects the current operand queue (not that of callers)
///
/// function will then encode the given instruction and run 1 step.
/// Then checks that the resulting state has the operand queue returned by
/// `expected_op_queue` (and that the program counter has incremented by 2). The
/// returned operand queue should not be advanced before being returned.
/// But, the ready list (index 0) must be removed. The queue will then  be
/// advanced before the check. If so, will then check that the reported metrics
/// match those returned by `expected_metrics` (plus that 1 instruction has been
/// read).
///
/// Both `expected_op_queue` and `expected_metrics` are given the current
/// function's operand queue before the step is performed.
fn test_simple_instruction(
	NoCF(state): NoCF<ExecState>,
	instr: Instruction,
	expected_op_queue: impl FnOnce(&OperandQueue) -> OperandQueue,
	expected_metrics: impl FnOnce(&OperandQueue) -> TrackReport,
	expected_events: impl FnOnce(&OperandQueue) -> OneShotTracker,
) -> TestResult
{
	// Build expected state
	let mut expected_state = state.clone();
	expected_state.address += 2;
	expected_state.frame.op_queue = expected_op_queue(&state.frame.op_queue);

	// Advance expected operand queue by 1
	assert!(expected_state.frame.op_queue.get(&0).is_none());
	expected_state.frame.op_queue = advance_queue(expected_state.frame.op_queue);

	// Build expected metrics
	let mut expected_mets = expected_metrics(&state.frame.op_queue);
	assert_eq!(expected_mets.get_stat(Metric::InstructionReads), 0);
	expected_mets.add_stat(Metric::InstructionReads, 1);

	// Build expected events
	let mut expected_evs = expected_events(&state.frame.op_queue);
	expected_evs.event(Event::Instruction(state.address, instr.clone()));

	test_execution_step(
		&state,
		RepeatingMem::<true>(instr.encode(), 0),
		&expected_state,
		&expected_mets,
		&expected_evs,
	)
}

/// For each operand in the queue at the given index, adds an event of the given
/// type.
///
/// Creates fresh IDs for each operand.
fn add_operand_events_for_list(
	evs: &mut OneShotTracker,
	queue: &OperandQueue,
	idx: usize,
	skip: usize,
	ev_tys: &[OperandEvent],
)
{
	if let Some(list) = queue.get(&idx)
	{
		for op in list
			.iter()
			.skip(skip)
			.map(|v| v.clone().into_operand(evs))
			.collect::<Vec<_>>()
		{
			for ev_ty in ev_tys
			{
				evs.event(Event::Id(op.id, IdEvent::Operand(*ev_ty)))
			}
		}
	}
}

/// For each operand in the queue at the given index, produces an event of the
/// given type.
///
/// Creates fresh IDs for each operand.
fn operand_events_for_list(
	queue: &OperandQueue,
	idx: usize,
	skip: usize,
	ev_tys: &[OperandEvent],
) -> OneShotTracker
{
	let mut evs = OneShotTracker::new();

	add_operand_events_for_list(&mut evs, queue, idx, skip, ev_tys);

	evs
}

/// Tests can convert from state to executor back to state without the state
/// changing
#[quickcheck]
fn import_export_executor(state: ExecState) -> bool
{
	let executor = Executor::from_state(&state, RepeatingMem::<true>(0, 0), &mut ());
	let new_state = executor.state();

	new_state == state
}

#[quickcheck]
fn max_operands_in_list(
	NoCF(mut state): NoCF<ExecState>,
	inst: Instruction,
	v: Value,
	rand: usize,
) -> TestResult
{
	if inst.out_refs().len() == 0
	{
		return TestResult::discard();
	}

	// Immediately triggering control flow may replace the current frame or
	// merge the target list into the caller's ready list, so the target list
	// cannot be expected to survive unchanged
	match inst
	{
		Instruction::Call(_, offset) | Instruction::Jump(_, offset) if offset.value == 0 =>
		{
			return TestResult::discard()
		},
		_ => (),
	}

	let out_ref = inst.out_refs()[rand % inst.out_refs().len()];

	// Fill the target operand list with values so that the instruction output
	// to it is discarded
	add_to_queue_idx(&mut state, out_ref + 1, repeat(v));
	let expected_list = state.frame.op_queue.get(&(out_ref + 1)).unwrap().clone();

	match std::panic::catch_unwind(|| {
		Executor::from_state(&state, RepeatingMem::<true>(inst.encode(), 0), &mut ())
			.step(&mut (), &mut ())
	})
	{
		Ok(Ok(exec)) =>
		{
			let actual_list = exec.state().frame.op_queue.get(&out_ref).unwrap().clone();
			if actual_list == expected_list
			{
				TestResult::passed()
			}
			else
			{
				TestResult::error(format!(
					"actual != expected:\n {:?}\n !=\n {:?}",
					actual_list, expected_list
				))
			}
		},
		_ => TestResult::discard(),
	}
}

/// Tests that the Nop instruction does nothing (discarding any operands)
#[quickcheck]
fn instruction_nop(state: NoCF<ExecState>) -> TestResult
{
	test_simple_instruction(
		state,
		Instruction::NoOp,
		|old_op_queue| {
			let mut new_op_q = old_op_queue.clone();
			// Discard ready list if present
			new_op_q.remove(&0);
			new_op_q
		},
		|_| [].into(),
		|old_op_queue| {
			let mut evs = OneShotTracker::new();
			if let Some(list) = old_op_queue.get(&0)
			{
				for _ in list.iter()
				{
					let id = evs.new_operand_id();
					evs.event(Event::Id(id, IdEvent::Operand(OperandEvent::Discard)));
				}
			}
			evs
		},
	)
}

/// Tests the Constant instruction
#[quickcheck]
fn instruction_constant(
	state: NoCF<ExecState>,
	typ_bits: Bits<3, false>,
	immediate: u8,
) -> TestResult
{
	let op_list_depth = state.0.op_list_depth();

	let typ: Type = typ_bits.try_into().unwrap();
	let mut bytes = vec![immediate];
	// The remaining bytes are sign extended if needed.
	bytes.resize(
		typ.size(),
		if typ.is_signed_int() && immediate >= 128
		{
			u8::MAX
		}
		else
		{
			0
		},
	);

	test_simple_instruction(
		state,
		Instruction::Constant(
			typ_bits,
			Bits {
				value: immediate as i32,
			},
		),
		|old_op_queue| {
			let mut new_op_q = old_op_queue.clone();

			// The produced operand should go first in the next operand list
			let old_operands = new_op_q.remove(&0);
			let new_const =
				Value::singleton_typed(typ.clone().into(), Scalar::Val(bytes.into_boxed_slice()));

			add_value_to_queue_idx(&mut new_op_q, op_list_depth, 1, new_const);

			if let Some(old_ops) = old_operands
			{
				add_list_to_queue_idx(&mut new_op_q, op_list_depth, 1, old_ops.into_iter());
			}
			new_op_q
		},
		|old_op_queue| {
			let old_op_count = old_op_queue.get(&0).map_or(0, |ops| ops.len());
			[
				(Metric::QueuedValues, 1),
				(Metric::QueuedValueBytes, typ.size()),
				(Metric::ReorderedOperands, old_op_count),
			]
			.into()
		},
		|old_op_queue| {
			let mut evs = OneShotTracker::new();

			let mut ops = vec![(evs.new_operand_id(), OperandEvent::Produce)];
			if let Some(list) = old_op_queue.get(&0)
			{
				ops.extend(
					list.iter()
						.map(|_| (evs.new_operand_id(), OperandEvent::Forward)),
				);
			}

			add_events_with_discards(
				&mut evs,
				ops,
				operand_queue_list_count(old_op_queue, 1),
				op_list_depth,
			);

			evs
		},
	)
}

/// Produces the events for a grow instruction.
///
/// The operands at the given ready queue will be assumed to follow some other
/// operand in the ready queue
fn grow_events(queue: &OperandQueue, depth: usize) -> OneShotTracker
{
	let mut evs = OneShotTracker::new();

	let consume_id = evs.new_operand_id();
	evs.event(Event::Id(
		consume_id,
		IdEvent::Operand(OperandEvent::Consume),
	));

	let mut ops = vec![(evs.new_operand_id(), OperandEvent::Produce)];
	if let Some(list) = queue.get(&0)
	{
		ops.extend(
			list.iter()
				.skip(1)
				.map(|_| (evs.new_operand_id(), OperandEvent::Forward)),
		);
	}

	add_events_with_discards(&mut evs, ops, operand_queue_list_count(queue, 1), depth);

	evs
}

/// Tests the Grow instruction on values
#[quickcheck]
fn instruction_grow(
	state: SimpleOps<LimitedOps<NoCF<ExecState>, 1, 4>>,
	immediate: u8,
) -> TestResult
{
	let op_list_depth = state.0 .0 .0.op_list_depth();
	let ready_q = state.0 .0 .0.frame.op_queue.get(&0).unwrap();
	let in_val = ready_q.first.clone();
	let mut result = in_val
		.first
		.bytes()
		.unwrap()
		.iter()
		.cloned()
		.take(in_val.typ.scale() - 1)
		.collect::<Vec<u8>>();
	result.insert(0, immediate);
	let result_value = Value::singleton_typed(in_val.typ, Scalar::Val(result.into_boxed_slice()));

	test_simple_instruction(
		state.0 .0.clone(),
		Instruction::Grow(Bits {
			value: immediate as i32,
		}),
		|old_op_queue| {
			let mut new_op_q = old_op_queue.clone();

			let ready_q = new_op_q.remove(&0).unwrap();

			// Put the produced value on the next queue with the rest
			add_list_to_queue_idx(
				&mut new_op_q,
				op_list_depth,
				1,
				once(result_value).chain(ready_q.rest.into_iter()),
			);

			new_op_q
		},
		|_| {
			[
				(Metric::QueuedValues, 1),
				(Metric::QueuedValueBytes, in_val.typ.scale()),
				(Metric::ConsumedOperands, 1),
				(Metric::ConsumedBytes, in_val.typ.scale()),
				(Metric::ReorderedOperands, ready_q.rest.len()),
			]
			.into()
		},
		|q| grow_events(q, op_list_depth),
	)
}

/// Tests growing the given value does not change it
fn instruction_grow_nop(
	mut state: LimitedOps<NoCF<ExecState>, 0, 3>,
	value: Value,
	immediate: u8,
) -> TestResult
{
	let op_list_depth = state.0 .0.op_list_depth();
	let ready_q_len = state.0 .0.frame.op_queue.get(&0).map_or(0, |l| l.len());

	let old_ops = state
		.0
		 .0
		.frame
		.op_queue
		.remove(&0)
		.map(|ops| ops.into_iter().collect())
		.unwrap_or(vec![]);
	add_to_queue_idx(
		&mut state.0 .0,
		0,
		once(value.clone()).chain(old_ops.into_iter()),
	);

	test_simple_instruction(
		state.0,
		Instruction::Grow(Bits {
			value: immediate as i32,
		}),
		|old_op_queue| {
			let mut new_op_q = old_op_queue.clone();

			let ready_q = new_op_q.remove(&0).unwrap();

			// Put the produced value on the next queue with the rest
			add_list_to_queue_idx(
				&mut new_op_q,
				op_list_depth,
				1,
				once(value.clone()).chain(ready_q.rest.into_iter()),
			);

			new_op_q
		},
		|_| {
			[
				(Metric::QueuedValues, 1),
				(Metric::QueuedValueBytes, value.typ.scale()),
				(Metric::ConsumedOperands, 1),
				(Metric::ConsumedBytes, value.typ.scale()),
				(
					Metric::ReorderedOperands,
					if ready_q_len >= op_list_depth
					{
						ready_q_len - 1
					}
					else
					{
						ready_q_len
					},
				),
			]
			.into()
		},
		|q| grow_events(q, op_list_depth),
	)
}

/// Tests the Grow instruction on NaNs
#[quickcheck]
fn instruction_grow_nan(
	state: LimitedOps<NoCF<ExecState>, 0, 3>,
	nan_type: ValueType,
	immediate: u8,
) -> TestResult
{
	instruction_grow_nop(state, Value::new_nan_typed(nan_type), immediate)
}

/// Tests the Grow instruction on NaRs
#[quickcheck]
fn instruction_grow_nar(
	state: LimitedOps<NoCF<ExecState>, 0, 3>,
	nar_type: ValueType,
	immediate: u8,
	nar_payload: usize,
) -> TestResult
{
	instruction_grow_nop(
		state,
		Value::new_nar_typed(nar_type, nar_payload),
		immediate,
	)
}
