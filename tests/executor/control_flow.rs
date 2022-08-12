use crate::{executor::SupportedInstruction, misc::RepeatingMem};
use quickcheck::TestResult;
use scry_isa::Instruction;
use scry_sim::{
	arbitrary::{InstrAddr, NoCF},
	CallFrameState, ControlFlowType, ExecError, ExecState, Executor, Metric, MetricTracker,
	OperandState, TrackReport,
};
use std::{collections::HashMap, iter::once};

#[quickcheck]
fn return_trigger(
	NoCF(state): NoCF<ExecState>,
	mut frame: CallFrameState,
	InstrAddr(ret_to): InstrAddr,
	instr: SupportedInstruction,
) -> TestResult
{
	let instr_encoded = Instruction::encode(&instr.0);

	// Ensure frame has no control flow after next instruction
	let _ = frame.branches.remove(&state.address);

	// Execute one step on the frame to get the expected result of the next
	// instruction
	let mut expected_metrics = TrackReport::new();
	let (mut expected_ready_ops, expected_reads) = match Executor::from_state(
		&ExecState {
			address: state.address,
			frame: frame.clone(),
			frame_stack: Vec::new(),
		},
		RepeatingMem(instr_encoded, 0),
	)
	.step(&mut expected_metrics)
	{
		Ok(exec) =>
		{
			let state = exec.state();
			let ready_ops = state.frame.op_queues.get(&0).cloned();
			let mut reads = Vec::new();
			let ready_ops = ready_ops.map(|(first, rest)| {
				let mut read_map = HashMap::new();
				let mut convert_read = |op: OperandState<usize>| {
					if let OperandState::MustRead(idx) = op
					{
						if let Some(idx_mapped) = read_map.get(&idx)
						{
							OperandState::MustRead(*idx_mapped)
						}
						else
						{
							reads.push(state.frame.reads[idx]);
							read_map.insert(idx, reads.len() - 1);
							OperandState::MustRead(reads.len() - 1)
						}
					}
					else
					{
						op
					}
				};
				(
					convert_read(first),
					rest.into_iter().map(|op| convert_read(op)).collect(),
				)
			});
			(ready_ops, reads)
		},
		Err(ExecError::Exception) => return TestResult::discard(),
		_ => return TestResult::discard(),
	};
	expected_metrics.add_stat(Metric::TriggeredReturns, 1);

	// Construct the expected end state
	let mut expected_state: ExecState = state.clone();
	expected_state.address = ret_to;

	// First, offset the expected operand's MustReads by the existing reads
	expected_ready_ops.iter_mut().for_each(
		|(first, rest): &mut (OperandState<usize>, Vec<OperandState<usize>>)| {
			once(first).chain(rest.iter_mut()).for_each(|op| {
				if let OperandState::MustRead(idx) = op
				{
					*idx += expected_state.frame.reads.len();
				}
			});
		},
	);

	// Add operands and reads to expected state
	if let Some((expected_first, expected_rest)) = expected_ready_ops
	{
		if let Some((_, rest)) = expected_state.frame.op_queues.get_mut(&0)
		{
			rest.push(expected_first);
			rest.extend(expected_rest);
		}
		else
		{
			expected_state
				.frame
				.op_queues
				.insert(0, (expected_first, expected_rest));
		}
		expected_state
			.frame
			.reads
			.extend(expected_reads.into_iter());
	}

	// Because the order of pending reads affects equality, but we don't
	// want to dictate the order, we insert the expected state into an Executor
	// and extract it immediately, ensuring the order of reads will follow the
	// implementation
	expected_state = Executor::from_state(&expected_state, RepeatingMem(0, 0)).state();

	// Construct test state
	let mut test_state = state.clone();
	frame.ret_addr = ret_to;
	frame
		.branches
		.insert(state.address, ControlFlowType::Return);
	let old_first_frame = std::mem::replace(&mut test_state.frame, frame);
	test_state.frame_stack.insert(0, old_first_frame);

	// Run test
	let mut actual_metrics = TrackReport::new();
	match Executor::from_state(&test_state, RepeatingMem(instr_encoded, 0))
		.step(&mut actual_metrics)
	{
		Ok(exec) =>
		{
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
				if expected_metrics != actual_metrics
				{
					TestResult::error(format!(
						"Unexpected step metrics (actual != expected):\n{:?} != {:?}",
						actual_metrics, expected_metrics
					))
				}
				else
				{
					TestResult::passed()
				}
			}
		},
		_ => TestResult::error("Test step failed"),
	}
}
