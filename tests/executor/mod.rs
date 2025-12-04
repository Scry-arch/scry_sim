use crate::misc::{advance_queue, RepeatingMem};
use quickcheck::{Arbitrary, Gen, TestResult};
use quickcheck_macros::quickcheck;
use scry_isa::{Alu2Variant, AluVariant, Bits, CallVariant, Instruction, Type};
use scry_sim::{
	arbitrary::{LimitedOps, NoCF, SimpleOps},
	ExecError, ExecState, Executor, Memory, Metric, MetricReporter, MetricTracker, OperandList,
	OperandQueue, Scalar, TrackReport, Value, ValueType,
};
use std::{borrow::BorrowMut, fmt::Debug, mem::replace};

mod alu_instructions;
mod cast;
mod control_flow;
mod data_flow;
mod load;
mod pick;
mod stack;
mod store;

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
pub fn test_metrics(expected: &TrackReport, actual: &TrackReport) -> TestResult
{
	if actual != expected
	{
		TestResult::error(format!(
			"Unexpected step metrics (actual != expected):\n{:?}\n !=\n {:?}",
			actual, expected
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
) -> TestResult
{
	let mut actual_metrics = TrackReport::new();
	match Executor::from_state(start_state, start_memory).step(&mut actual_metrics)
	{
		Ok(exec) =>
		{
			if exec.state() != *expected_state
			{
				TestResult::error(format!(
					"Unexpected end state (actual != expected):\n{:?}\n !=\n {:?}",
					exec.state(),
					expected_state
				))
			}
			else
			{
				test_metrics(expected_metrics, &actual_metrics)
			}
		},
		err => TestResult::error(format!("Test step failed: {:?}", err)),
	}
}

/// Tests that an execution step from the given state/memory results in an
/// exception with the given expected metrics.
pub fn test_execution_step_exceptions<M: Memory + Debug>(
	start_state: &ExecState,
	start_memory: impl BorrowMut<M> + Debug,
	expected_metrics: &TrackReport,
) -> TestResult
{
	let mut actual_metrics = TrackReport::new();
	match Executor::from_state(start_state, start_memory).step(&mut actual_metrics)
	{
		Err(ExecError::Exception(_)) => test_metrics(expected_metrics, &actual_metrics),
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

	test_execution_step(
		&state,
		RepeatingMem::<true>(instr.encode(), 0),
		&expected_state,
		&expected_mets,
	)
}

/// Tests can convert from state to executor back to state without the state
/// changing
#[quickcheck]
fn import_export_executor(state: ExecState) -> bool
{
	let executor = Executor::from_state(&state, RepeatingMem::<true>(0, 0));
	let new_state = executor.state();

	new_state == state
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

			let ops_rest = if let Some(ops) = new_op_q.get_mut(&1)
			{
				ops.push(new_const);
				ops
			}
			else
			{
				new_op_q.insert(1, OperandList::new(new_const, Vec::new()));
				new_op_q.get_mut(&1).unwrap()
			};
			if let Some(old_ops) = old_operands
			{
				ops_rest.extend(old_ops.into_iter());
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
	)
}

/// Tests the Grow instruction on values
#[quickcheck]
fn instruction_grow(
	state: SimpleOps<LimitedOps<NoCF<ExecState>, 1, 4>>,
	immediate: u8,
) -> TestResult
{
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
			if let Some(q) = new_op_q.get_mut(&1)
			{
				q.rest.push(result_value.clone());
				q.rest.extend(ready_q.rest.into_iter());
			}
			else
			{
				new_op_q.insert(1, OperandList::new(result_value.clone(), ready_q.rest));
			}

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
	)
}

/// Tests the Grow instruction on values
#[quickcheck]
fn instruction_grow_nan(
	mut state: LimitedOps<NoCF<ExecState>, 0, 3>,
	nan_type: ValueType,
	immediate: u8,
) -> TestResult
{
	let ready_q_len = state.0 .0.frame.op_queue.get(&0).map_or(0, |l| l.len());
	let result_value = Value::new_nan_typed(nan_type);

	if let Some(q) = state.0 .0.frame.op_queue.get_mut(&0)
	{
		let old_first = replace(&mut q.first, result_value.clone());
		q.rest.insert(0, old_first);
	}
	else
	{
		state
			.0
			 .0
			.frame
			.op_queue
			.insert(0, OperandList::new(result_value.clone(), Vec::new()));
	}

	test_simple_instruction(
		state.0,
		Instruction::Grow(Bits {
			value: immediate as i32,
		}),
		|old_op_queue| {
			let mut new_op_q = old_op_queue.clone();

			let ready_q = new_op_q.remove(&0).unwrap();

			// Put the produced value on the next queue with the rest
			if let Some(q) = new_op_q.get_mut(&1)
			{
				q.rest.push(result_value.clone());
				q.rest.extend(ready_q.rest.into_iter());
			}
			else
			{
				new_op_q.insert(1, OperandList::new(result_value.clone(), ready_q.rest));
			}

			new_op_q
		},
		|_| {
			[
				(Metric::QueuedValues, 1),
				(Metric::QueuedValueBytes, nan_type.scale()),
				(Metric::ConsumedOperands, 1),
				(Metric::ConsumedBytes, nan_type.scale()),
				(Metric::ReorderedOperands, ready_q_len),
			]
			.into()
		},
	)
}

/// Tests the Grow instruction on values
#[quickcheck]
fn instruction_grow_nar(
	mut state: LimitedOps<NoCF<ExecState>, 0, 3>,
	nar_type: ValueType,
	immediate: u8,
	nar_payload: usize,
) -> TestResult
{
	let ready_q_len = state.0 .0.frame.op_queue.get(&0).map_or(0, |l| l.len());
	let result_value = Value::new_nar_typed(nar_type, nar_payload);

	if let Some(q) = state.0 .0.frame.op_queue.get_mut(&0)
	{
		let old_first = replace(&mut q.first, result_value.clone());
		q.rest.insert(0, old_first);
	}
	else
	{
		state
			.0
			 .0
			.frame
			.op_queue
			.insert(0, OperandList::new(result_value.clone(), Vec::new()));
	}

	test_simple_instruction(
		state.0,
		Instruction::Grow(Bits {
			value: immediate as i32,
		}),
		|old_op_queue| {
			let mut new_op_q = old_op_queue.clone();

			let ready_q = new_op_q.remove(&0).unwrap();

			// Put the produced value on the next queue with the rest
			if let Some(q) = new_op_q.get_mut(&1)
			{
				q.rest.push(result_value.clone());
				q.rest.extend(ready_q.rest.into_iter());
			}
			else
			{
				new_op_q.insert(1, OperandList::new(result_value.clone(), ready_q.rest));
			}

			new_op_q
		},
		|_| {
			[
				(Metric::QueuedValues, 1),
				(Metric::QueuedValueBytes, nar_type.scale()),
				(Metric::ConsumedOperands, 1),
				(Metric::ConsumedBytes, nar_type.scale()),
				(Metric::ReorderedOperands, ready_q_len),
			]
			.into()
		},
	)
}
