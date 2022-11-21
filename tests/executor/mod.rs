use crate::misc::{advance_queue, RepeatingMem};
use quickcheck::{Arbitrary, Gen, TestResult};
use quickcheck_macros::quickcheck;
use scry_isa::{Alu2Variant, AluVariant, Bits, BitsDyn, CallVariant, Instruction};
use scry_sim::{
	arbitrary::NoCF, ExecError, ExecState, Executor, Memory, Metric, MetricReporter, MetricTracker,
	OperandList, OperandQueue, OperandState, TrackReport,
};
use std::{borrow::BorrowMut, fmt::Debug};

mod alu_instructions;
mod control_flow;
mod data_flow;
mod load;
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
				| Alu(AluVariant::Inc, _)
				| Alu(AluVariant::Sub, _)
				| Alu(AluVariant::Dec, _)
				| Alu2(Alu2Variant::Add, _, _)
				| Alu2(Alu2Variant::Sub, _, _)
				| Constant(..)
				| Call(CallVariant::Ret, _)
				| Duplicate(..)
				| Echo(..)
				| EchoLong(..)
				| Capture(..) => break,
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

/// Used to generate arbitrary instruction that are supported by the
/// executor and consume or discard all their inputs
#[derive(Debug, Clone)]
struct ConsumingDiscarding(Instruction);
impl ConsumingDiscarding
{
	// Returns whether an operand with the given index in the ready list
	// is discarded by this instruction.
	fn discards(&self, idx: usize) -> bool
	{
		use Instruction::*;
		idx >= match self.0
		{
			Capture(..) => 0,
			Alu(AluVariant::Inc, _) | Alu(AluVariant::Dec, _) => 1,
			Alu(AluVariant::Add, _)
			| Alu(AluVariant::Sub, _)
			| Alu2(Alu2Variant::Add, _, _)
			| Alu2(Alu2Variant::Sub, _, _) => 2,
			_ => panic!("Other instructions aren't guaranteed to consume or discard their inputs."),
		}
	}
}
impl Arbitrary for ConsumingDiscarding
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
				| Alu(AluVariant::Inc, _)
				| Alu(AluVariant::Sub, _)
				| Alu(AluVariant::Dec, _)
				| Alu2(Alu2Variant::Add, _, _)
				| Alu2(Alu2Variant::Sub, _, _) => break,
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
		Err(ExecError::Exception) => test_metrics(expected_metrics, &actual_metrics),
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
	// Ensure any superfluous reads are removed
	expected_state.clean_reads();

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
		Instruction::nop(),
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
fn instruction_constant(state: NoCF<ExecState>, immediate: BitsDyn<8>) -> TestResult
{
	test_simple_instruction(
		state,
		Instruction::Constant(immediate.clone()),
		|old_op_queue| {
			let mut new_op_q = old_op_queue.clone();

			// The produced operand should go first in the next operand list
			let old_operands = new_op_q.remove(&0);
			let new_const = OperandState::Ready(
				if immediate.is_signed()
				{
					(Bits::<8, true>::try_from(immediate).unwrap().value as i8).into()
				}
				else
				{
					(Bits::<8, false>::try_from(immediate).unwrap().value as u8).into()
				},
			);

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
				(Metric::QueuedValueBytes, 1),
				(Metric::ReorderedOperands, old_op_count),
			]
			.into()
		},
	)
}
