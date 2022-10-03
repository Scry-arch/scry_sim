use crate::misc::{advance_queues, RepeatingMem};
use quickcheck::{Arbitrary, Gen, TestResult};
use scry_isa::{Alu2Variant, AluVariant, Bits, BitsDyn, CallVariant, Instruction};
use scry_sim::{
	arbitrary::NoCF, ExecState, Executor, Metric, MetricReporter, MetricTracker, OperandState,
	TrackReport,
};
use std::collections::HashMap;

mod alu_instructions;
mod control_flow;
mod data_flow;

/// Used to generate arbitrary instruction that are supported by the
/// executor
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
				| Nop
				| Constant(_)
				| Call(CallVariant::Ret, _)
				| Duplicate(..) => break,
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

/// Tests a "simple" instruction, which must not:
///
/// * Issue control flow
/// * Consume operands (may discard or reorder)
/// * Only affects the current operand queue (not that of callers)
///
/// function will then encode the given instruction and run 1 step.
/// Then checks that the resulting state is has the operand queue returned
/// by `expected_op_queues` (and that the program counter has incremented by 2).
/// The returned operand queue should not be advanced before being returned.
/// But, the ready queue (index 0) must be removed. The queue will then  be
/// advanced before the check. If so, will then check that the reported metrics
/// match those returned by `expected_metrics` (plus that 1 instruction has been
/// read).
///
/// Both `expected_op_queues` and `expected_metrics` are given the current
/// function's operand queue before the step is performed.
fn test_simple_instruction(
	NoCF(state): NoCF<ExecState>,
	instr: Instruction,
	expected_op_queues: impl FnOnce(
		&HashMap<usize, (OperandState<usize>, Vec<OperandState<usize>>)>,
	) -> HashMap<usize, (OperandState<usize>, Vec<OperandState<usize>>)>,
	expected_metrics: impl FnOnce(
		&HashMap<usize, (OperandState<usize>, Vec<OperandState<usize>>)>,
	) -> TrackReport,
) -> TestResult
{
	let exec = Executor::from_state(&state, RepeatingMem(instr.encode(), 0));
	let mut metrics = TrackReport::new();
	match exec.step(&mut metrics)
	{
		Ok(exec) =>
		{
			let mut expected_state = state.clone();
			expected_state.address += 2;
			expected_state.frame.op_queues = expected_op_queues(&state.frame.op_queues);

			// Advance expected operand queue by 1
			assert!(expected_state.frame.op_queues.get(&0).is_none());
			expected_state.frame.op_queues = advance_queues(expected_state.frame.op_queues);
			// Ensure any superfluous reads are removed
			expected_state.clean_reads();

			if exec.state() != expected_state
			{
				TestResult::error(format!(
					"Unexpected end state (actual != expected):\n{:?} != {:?}",
					exec.state(),
					expected_state
				))
			}
			else
			{
				let mut expected_mets = expected_metrics(&state.frame.op_queues);
				assert_eq!(expected_mets.get_stat(Metric::InstructionReads), 0);
				expected_mets.add_stat(Metric::InstructionReads, 1);

				if metrics != expected_mets
				{
					TestResult::error(format!(
						"Unexpected step metrics (actual != expected):\n{:?} != {:?}",
						metrics, expected_mets
					))
				}
				else
				{
					TestResult::from_bool(true)
				}
			}
		},
		err => TestResult::error(format!("Test step failed: {:?}", err)),
	}
}

/// Tests can convert from state to executor back to state without the state
/// changing
#[quickcheck]
fn import_export_executor(state: ExecState) -> bool
{
	let executor = Executor::from_state(&state, RepeatingMem(0, 0));
	let new_state = executor.state();

	new_state == state
}

/// Tests that the Nop instruction does nothing (discarding any operands)
#[quickcheck]
fn instruction_nop(state: NoCF<ExecState>) -> TestResult
{
	test_simple_instruction(
		state,
		Instruction::Nop,
		|old_op_queues| {
			let mut new_op_q = old_op_queues.clone();
			// Discard ready-queue if present
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
		|old_op_queues| {
			let mut new_op_q = old_op_queues.clone();

			// The produced operand should go first in the next operands queue
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

			let ops_rest = if let Some((_, rest)) = new_op_q.get_mut(&1)
			{
				rest.push(new_const);
				rest
			}
			else
			{
				new_op_q.insert(1, (new_const, Vec::new()));
				&mut new_op_q.get_mut(&1).unwrap().1
			};
			if let Some((old_first, old_rest)) = old_operands
			{
				ops_rest.push(old_first);
				ops_rest.extend(old_rest.into_iter());
			}
			new_op_q
		},
		|old_op_queues| {
			let old_op_count = old_op_queues.get(&0).map_or(0, |(_, rest)| 1 + rest.len());
			[
				(Metric::QueuedValues, 1),
				(Metric::QueuedValueBytes, 1),
				(Metric::ReorderedOperands, old_op_count),
			]
			.into()
		},
	)
}
