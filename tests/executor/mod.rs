use crate::misc::RepeatingMem;
use byteorder::{ByteOrder, LittleEndian};
use quickcheck::{Arbitrary, Gen, TestResult};
use scry_isa::{Alu2Variant, AluVariant, Instruction};
use scryer::{
	arbitrary::NoCF, execution::Executor, memory::BlockedMemory, ExecState, Metric, TrackReport,
};

mod alu_instructions;
mod control_flow;
/// Tests can convert from state to executor back to state without the state
/// changing
#[quickcheck]
fn import_export_executor(state: ExecState) -> bool
{
	let executor = Executor::from_state(&state, RepeatingMem(0, 0));
	let new_state = executor.state();

	new_state == state
}

#[quickcheck]
fn instruction_nop(NoCF(state): NoCF<ExecState>) -> TestResult
{
	let mut nop_encoded = [0; 2];
	LittleEndian::write_u16(&mut nop_encoded, Instruction::Nop.encode());
	let exec = Executor::from_state(
		&state,
		BlockedMemory::new(nop_encoded.into(), state.address),
	);
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
					TestResult::error(format!("{:?} != {:?}", metrics, expected_mets))
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
				// | Call(CallVariant::Ret, _)
				=> break,
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
