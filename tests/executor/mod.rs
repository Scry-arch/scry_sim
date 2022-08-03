use byteorder::{ByteOrder, LittleEndian};
use quickcheck::{Arbitrary, Gen, TestResult};
use scry_isa::{Alu2Variant, AluVariant, CallVariant, Instruction};
use scryer::{
	arbitrary::NoCF,
	execution::{ExecResult, Executor},
	memory::BlockedMemory,
	ExecState, Metric, OperandState, TrackReport,
};
use std::iter::once;

mod alu_instructions;
mod control_flow;
/// Tests can convert from state to executor back to state without the state
/// changing
#[quickcheck]
fn import_export_executor(state: ExecState, mem: Vec<u8>, base_addr: usize) -> bool
{
	let executor = Executor::from_state(&state, BlockedMemory::new(mem, base_addr));
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
		ExecResult::Ok(exec) =>
		{
			let mut expected_state = state.clone();
			expected_state.address += 2;
			let mut new_op_q = state.frame.op_queues.clone();
			// Discard ready-queue if present
			let discarded = new_op_q.remove(&0);
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
				// Calculate expected discard metrics
				let (disc_val, disc_bytes, disc_reads) =
					discarded.map_or((0, 0, 0), |(op1, op_rest)| {
						once(&op1)
							.chain(op_rest.iter())
							.fold((0, 0, 0), |mut acc, op| {
								match op
								{
									OperandState::Ready(v) =>
									{
										acc.0 += 1;
										acc.1 += v.size();
									},
									_ => acc.2 += 1,
								}
								acc
							})
					});

				// Check metrics
				let expected_mets: TrackReport = [
					(Metric::DiscardedValues, disc_val),
					(Metric::DiscardedValuesBytes, disc_bytes),
					(Metric::DiscardedReads, disc_reads),
					(Metric::InstructionReads, 1),
				]
				.into();

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
