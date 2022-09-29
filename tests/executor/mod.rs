use crate::misc::{clone_advance_queues, RepeatingMem};
use quickcheck::{Arbitrary, Gen, TestResult};
use scry_isa::{Alu2Variant, AluVariant, Bits, BitsDyn, CallVariant, Instruction};
use scry_sim::{arbitrary::NoCF, ExecState, Executor, Metric, OperandState, TrackReport};

mod alu_instructions;
mod control_flow;

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
				| Call(CallVariant::Ret, _) => break,
				_ => instr = Instruction::arbitrary(g),
			}
		}
		Self(instr)
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		dbg!(&self);
		Box::new(self.0.shrink().map(|shrunk| Self(shrunk)))
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
fn instruction_nop(NoCF(state): NoCF<ExecState>) -> TestResult
{
	let exec = Executor::from_state(&state, RepeatingMem(Instruction::Nop.encode(), 0));
	let mut metrics = TrackReport::new();
	match exec.step(&mut metrics)
	{
		Ok(exec) =>
		{
			let mut expected_state = state.clone();
			expected_state.address += 2;
			let mut new_op_q = state.frame.op_queues.clone();
			// Discard ready-queue if present
			new_op_q.remove(&0);
			// Reduce all queue indexes by 1
			expected_state.frame.op_queues = new_op_q
				.into_iter()
				.map(|(idx, ops)| (idx - 1, ops))
				.collect();
			// Ensure any superfluous reads are removed
			expected_state.clean_reads();

			if exec.state() != expected_state
			{
				TestResult::error(format!("{:?} != {:?}", exec.state(), expected_state))
			}
			else
			{
				// Check metrics
				let expected_mets: TrackReport = [(Metric::InstructionReads, 1)].into();

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
		err => TestResult::error(format!("Unexpected: {:?}", err)),
	}
}

/// Tests the Constant instruction
#[quickcheck]
fn instruction_constant(NoCF(state): NoCF<ExecState>, immediate: BitsDyn<8>) -> TestResult
{
	let exec = Executor::from_state(
		&state,
		RepeatingMem(Instruction::Constant(immediate.clone()).encode(), 0),
	);
	let mut metrics = TrackReport::new();
	match exec.step(&mut metrics)
	{
		Ok(exec) =>
		{
			let mut expected_state = state.clone();
			expected_state.address += 2;
			let mut new_op_q = state.frame.op_queues.clone();

			// The produced operands should go first in the next operands queue
			let old_operands = new_op_q.remove(&0);
			let old_op_count = old_operands.as_ref().map_or(0, |(_, rest)| 1 + rest.len());
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

			expected_state.frame.op_queues = clone_advance_queues(&new_op_q);

			if exec.state() != expected_state
			{
				TestResult::error(format!("{:?} != {:?}", exec.state(), expected_state))
			}
			else
			{
				// Check metrics
				let expected_mets: TrackReport = [
					(Metric::InstructionReads, 1),
					(Metric::QueuedValues, 1),
					(Metric::QueuedValueBytes, 1),
					(Metric::ReorderedOperands, old_op_count),
				]
				.into();

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
		err => TestResult::error(format!("Unexpected: {:?}", err)),
	}
}
