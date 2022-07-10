use byteorder::{ByteOrder, LittleEndian};
use quickcheck::TestResult;
use scry_isa::Instruction;
use scryer::{
	arbitrary::NoCFExecState,
	execution::{ExecResult, Executor},
	memory::Memory,
	ExecState,
};

mod control_flow;

/// Tests can convert from state to executor back to state without the state
/// changing
#[quickcheck]
fn import_export_executor(state: ExecState, mem: Vec<u8>, base_addr: usize) -> bool
{
	let executor = Executor::from_state(&state, Memory::new(mem, base_addr));
	let new_state = executor.state();

	new_state == state
}

#[quickcheck]
fn nop_discards_input(NoCFExecState(state): NoCFExecState) -> TestResult
{
	let mut nop_encoded = [0; 2];
	LittleEndian::write_u16(&mut nop_encoded, Instruction::Nop.encode());
	let exec = Executor::from_state(&state, Memory::new(nop_encoded.into(), state.address));

	match exec.step()
	{
		ExecResult::Ok(exec) =>
		{
			let mut expected_state = state.clone();
			expected_state.address += 2;
			let mut new_op_q = state.frame.op_queues.clone();
			// Discard ready-queue if present
			let _ = new_op_q.remove(&0);
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
				TestResult::from_bool(true)
			}
		},
		err => TestResult::error(format!("Unexpected: {:?}", err)),
	}
}
