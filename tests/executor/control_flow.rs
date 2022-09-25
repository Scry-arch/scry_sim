use crate::{
	executor::SupportedInstruction,
	misc::{clone_advance_queues, RepeatingMem},
};
use quickcheck::TestResult;
use scry_isa::{BitValue, Bits, CallVariant, Instruction};
use scry_sim::{
	arbitrary::NoCF, CallFrameState, ControlFlowType, ExecError, ExecState, Executor, Metric,
	MetricTracker, OperandState, TrackReport, ValueType,
};
use std::{collections::HashMap, iter::once};

/// Used to test triggering of returns
fn return_trigger_impl(
	// The execution state excluding the current call frame.
	// Must not include a control-flow trigger in the next instruction
	// since they are inactive frames and therefore would never include
	// one.
	NoCF(state): NoCF<ExecState>,
	// The executing call frame
	mut frame: CallFrameState,
	// The next instruction to be executed
	instr: Instruction,
	// Operands expected to be at the end of the ready queue after the return trigger.
	// They should be following any operands that were already there from before the call.
	mut expected_ready_ops: Option<(OperandState<usize>, Vec<OperandState<usize>>)>,
	// The reads of the expected operands
	expected_reads: Vec<(usize, usize, ValueType)>,
	// The metrics expected after the execution step
	expected_metrics: TrackReport,
) -> TestResult
{
	let instr_encoded = Instruction::encode(&instr);

	// Construct the expected end state
	let mut expected_state: ExecState = state.clone();
	expected_state.address = frame.ret_addr;

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

/// Test the triggering of a return that was previously issued.
#[quickcheck]
fn return_trigger(
	NoCF(state): NoCF<ExecState>,
	mut frame: CallFrameState,
	instr: SupportedInstruction,
) -> TestResult
{
	if let Instruction::Call(CallVariant::Ret, _) = instr.0
	{
		return TestResult::discard();
	}
	let instr_encoded = Instruction::encode(&instr.0);

	// Ensure frame has no control flow after next instruction
	let _ = frame.branches.remove(&state.address);

	// Execute one step on the frame to get the expected result of the next
	// instruction
	let mut expected_metrics = TrackReport::new();
	let (expected_ready_ops, expected_reads) = match Executor::from_state(
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

	return_trigger_impl(
		NoCF(state),
		frame,
		instr.0,
		expected_ready_ops,
		expected_reads,
		expected_metrics,
	)
}

/// Test return instructions with 0-offset (return immediately)
#[quickcheck]
fn return_immediately(state: NoCF<ExecState>, frame: CallFrameState) -> TestResult
{
	// The operands given to the branch location must be moved into the caller
	// frame.
	let mut expected_ready_ops = frame.op_queues.get(&1).cloned();
	let mut expected_reads = Vec::new();
	if let Some((first, rest)) = &mut expected_ready_ops
	{
		let mut read_map = Vec::new();
		std::iter::once(first).chain(rest).for_each(|op| {
			if let OperandState::MustRead(read_idx) = op
			{
				// Read operands must be renumbered
				if !read_map.contains(read_idx)
				{
					read_map.push(*read_idx);
					expected_reads.push(frame.reads[*read_idx]);
				}
				*read_idx = read_map
					.iter()
					.enumerate()
					.find(|(_, old_idx)| read_idx == *old_idx)
					.unwrap()
					.0;
			}
		});
	}

	return_trigger_impl(
		state,
		frame,
		Instruction::Call(CallVariant::Ret, 0.try_into().unwrap()),
		expected_ready_ops,
		expected_reads,
		TrackReport::from([
			(Metric::IssuedReturns, 1),
			(Metric::TriggeredReturns, 1),
			(Metric::InstructionReads, 1),
		]),
	)
}

/// Test the return instruction that doesn't immediately trigger.
#[quickcheck]
fn return_non_trigger(NoCF(state): NoCF<ExecState>, offset: Bits<6, false>) -> TestResult
{
	if offset.value() == 0
	{
		return TestResult::discard();
	}

	let instr_encoded = Instruction::encode(&Instruction::Call(CallVariant::Ret, offset));

	let mut expected_state = state.clone();
	// Return discards any operands its given
	expected_state.frame.op_queues.remove(&0);
	expected_state.frame.clean_reads();
	expected_state.frame.op_queues = clone_advance_queues(&expected_state.frame.op_queues);
	expected_state.frame.branches.insert(
		state.address + (offset.value() as usize * 2),
		ControlFlowType::Return,
	);
	expected_state.address += 2;

	// Run test
	let mut actual_metrics = TrackReport::new();
	match Executor::from_state(&state, RepeatingMem(instr_encoded, 0)).step(&mut actual_metrics)
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
				use Metric::*;
				let expected_metrics =
					TrackReport::from([(IssuedReturns, 1), (InstructionReads, 1)]);
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
