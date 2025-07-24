use crate::{
	executor::{test_execution_step, SupportedInstruction},
	misc::{
		advance_queue, get_absolute_address, get_relative_address, regress_queue, RepeatingMem,
	},
};
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_isa::{BitValue, Bits, CallVariant, Instruction};
use scry_sim::{
	arbitrary::{ArbValue, InstrAddr, NoCF},
	Block, CallFrameState, ControlFlowType, ExecState, Executor, Metric,
	Metric::{ConsumedBytes, ConsumedOperands, IssuedCalls},
	MetricTracker, OperandList, OperandState, Scalar, StackFrame, TrackReport, ValueType,
};
use std::collections::HashMap;

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
	// Operands expected to be at the end of the ready list after the return trigger.
	// They should be following any operands that were already there from before the call.
	mut expected_ready_ops: Option<OperandList>,
	// The reads of the expected operands
	expected_reads: Vec<(bool, usize, usize, ValueType)>,
	// The metrics expected after the execution step
	expected_metrics: TrackReport,
) -> TestResult
{
	let instr_encoded = Instruction::encode(&instr);

	// Construct the expected end state
	let mut expected_state: ExecState = state.clone();
	expected_state.address = frame.ret_addr;

	// First, offset the expected operand's MustReads by the existing reads
	expected_ready_ops.iter_mut().for_each(|ops| {
		ops.iter_mut().for_each(|op| {
			if let OperandState::MustRead(idx) = op
			{
				*idx += expected_state.frame.reads.len();
			}
		});
	});

	// Add operands and reads to expected state
	if let Some(expected_ops) = expected_ready_ops
	{
		if let Some(op_list) = expected_state.frame.op_queue.get_mut(&0)
		{
			expected_ops.into_iter().for_each(|op| op_list.push(op));
		}
		else
		{
			expected_state.frame.op_queue.insert(0, expected_ops);
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
	expected_state = Executor::from_state(&expected_state, RepeatingMem::<true>(0, 0)).state();
	expected_state.frame.stack.block.size += frame.stack.base_size;

	// Construct test state
	let mut test_state = state.clone();
	frame
		.branches
		.insert(state.address, ControlFlowType::Return);
	let old_first_frame = std::mem::replace(&mut test_state.frame, frame);
	test_state.frame_stack.insert(0, old_first_frame);

	test_execution_step(
		&test_state,
		RepeatingMem::<true>(instr_encoded, 0),
		&expected_state,
		&expected_metrics,
	)
}

/// Test the triggering of a return that was previously issued.
#[quickcheck]
fn return_trigger(
	NoCF(state): NoCF<ExecState>,
	mut frame: CallFrameState,
	instr: SupportedInstruction,
) -> TestResult
{
	// Don't test the actually return instruction itself
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
			stack_buffer: 0,
		},
		RepeatingMem::<true>(instr_encoded, 0),
	)
	.step(&mut expected_metrics)
	{
		Ok(exec) =>
		{
			let state = exec.state();
			let ready_ops = state.frame.op_queue.get(&0).cloned();
			let mut reads = Vec::new();
			let ready_ops = ready_ops.map(|mut op_list| {
				let mut read_map = HashMap::<usize, _>::new();
				op_list.iter_mut().for_each(|op| {
					if let OperandState::MustRead(idx) = op
					{
						if let Some(idx_mapped) = read_map.get(idx)
						{
							*idx = *idx_mapped;
						}
						else
						{
							reads.push(state.frame.reads[*idx]);
							read_map.insert(*idx, reads.len() - 1);
							*idx = reads.len() - 1;
						}
					}
				});
				op_list
			});
			(ready_ops, reads)
		},
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
	let mut expected_ready_ops = frame.op_queue.get(&1).cloned();
	let mut expected_reads = Vec::new();
	if let Some(ops) = &mut expected_ready_ops
	{
		let mut read_map = Vec::new();
		ops.iter_mut().for_each(|op| {
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
	expected_state.frame.op_queue.remove(&0);
	expected_state.frame.clean_reads();
	expected_state.frame.op_queue = advance_queue(expected_state.frame.op_queue);
	expected_state.frame.branches.insert(
		state.address + (offset.value() as usize * 2),
		ControlFlowType::Return,
	);
	expected_state.address += 2;

	use Metric::*;
	let expected_metrics = TrackReport::from([(IssuedReturns, 1), (InstructionReads, 1)]);

	test_execution_step(
		&state,
		RepeatingMem::<true>(instr_encoded, 0),
		&expected_state,
		&expected_metrics,
	)
}

/// Test the triggering of a jump that was previously issued.
#[quickcheck]
fn jmp_trigger(
	NoCF(state): NoCF<ExecState>,
	instr: SupportedInstruction,
	branch_target: InstrAddr,
) -> TestResult
{
	// Skip any test with immediate control flow triggering or trigger before the
	// target address or where the instruction is a branch
	match instr.0
	{
		Instruction::Call(_, offset) =>
		{
			if offset.value == 0
				|| ((offset.value() * 2) as usize + state.address) < branch_target.0
			{
				return TestResult::discard();
			}
		},
		Instruction::Jump(..) => return TestResult::discard(),
		_ => (),
	}
	// Discard test if jumping past trigger location
	if state
		.frame
		.branches
		.iter()
		.find(|(k, _)| **k < branch_target.0)
		.is_some()
	{
		return TestResult::discard();
	}

	let instr_encoded = Instruction::encode(&instr.0);

	// Execute one step on the frame to get the expected result of the next
	// instruction
	let mut expected_metrics = TrackReport::new();
	let mut expected_state =
		match Executor::from_state(&state, RepeatingMem::<true>(instr_encoded, 0))
			.step(&mut expected_metrics)
		{
			Ok(exec) => exec.state(),
			_ => return TestResult::discard(),
		};
	expected_metrics.add_stat(Metric::TriggeredBranches, 1);

	// Move the current address to the branch target
	expected_state.address = branch_target.0;

	// Construct test state
	let mut test_state = state.clone();
	test_state
		.frame
		.branches
		.insert(state.address, ControlFlowType::Branch(branch_target.0));

	test_execution_step(
		&test_state,
		RepeatingMem::<true>(instr_encoded, 0),
		&expected_state,
		&expected_metrics,
	)
}

/// Test the triggering of a jump that was previously issued.
#[quickcheck]
fn jmp_trigger_past_trigger(
	NoCF(state): NoCF<ExecState>,
	instr: SupportedInstruction,
	branch_target: InstrAddr,
	trigger_offset: usize,
) -> TestResult
{
	// Skip any test with immediate control flow triggering
	// or where the instruction is a branch
	match instr.0
	{
		Instruction::Call(_, offset) if offset.value == 0 => return TestResult::discard(),
		Instruction::Jump(..) => return TestResult::discard(),
		_ => (),
	}
	let instr_encoded = Instruction::encode(&instr.0);

	// Construct trigger address earlier than the target but after current address
	let range = if state.address < branch_target.0
	{
		branch_target.0 - state.address
	}
	else
	{
		branch_target.0
	};
	if range <= 1
	{
		return TestResult::discard();
	}
	let trig = branch_target.0 - (((trigger_offset % (range / 2)) + 1) * 2);

	// Construct test state
	let mut test_state = state.clone();
	test_state
		.frame
		.branches
		.insert(trig, ControlFlowType::Branch(0)); // target doesnt matter

	// Add the new branch trigger that jumps past the previous trigger
	test_state
		.frame
		.branches
		.insert(state.address, ControlFlowType::Branch(branch_target.0));

	TestResult::from_bool(
		Executor::from_state(&test_state, RepeatingMem::<true>(instr_encoded, 0))
			.step(&mut ())
			.is_err(),
	)
}

/// Test the jump instruction when taking 1 operand (i.e. immediate target and
/// location offsets).
///
/// Takes the given start state and jump target, location, and condition.
/// If no condition is given, will test the unconditional variant of the jump
/// (no operands). Tests that the state almost doesn't change expect the
/// following:
///
/// * If the jump is taken, `expect_jump` should alter the expected state given
///   to it
/// using the target and location addresses given.
/// * If `may_trigger` is true, the expected state's address is only incremented
///   by 2
/// if the jump is not taken, otherwise it is always incremented.
fn test_jump_immediate(
	NoCF(state): NoCF<ExecState>,
	target: Bits<7, true>,
	location: Bits<6, false>,
	condition: Option<ArbValue<false, false>>,
	expect_jump: impl FnOnce(&mut ExecState, usize, usize),
	may_trigger: bool,
) -> TestResult
{
	// Construct test state
	let mut test_state: ExecState = state.clone();
	// Regress operand queue so that we can put our input as the
	// the ready list.
	test_state.frame.op_queue = regress_queue(test_state.frame.op_queue);
	if let Some(condition) = &condition
	{
		test_state.frame.op_queue.insert(
			0,
			OperandList::new(OperandState::Ready(condition.0.clone()), vec![]),
		);
	}

	// Expected state should be the same as the given one (since we first regressed
	// it for our test.), except the address
	let mut expected_state: ExecState = state.clone();
	if !may_trigger
	{
		expected_state.address += 2;
	}

	let (unconditional, is_zero) = if let Some(condition) = &condition
	{
		if let Scalar::Val(s) = condition.0.get_first()
		{
			(false, s.iter().all(|b| *b == 0))
		}
		else
		{
			unreachable!()
		}
	}
	else
	{
		(true, false)
	};
	let location_addr = if let Some(addr) = state.address.checked_add((location.value * 2) as usize)
	{
		addr
	}
	else
	{
		// Undefined behavior
		return TestResult::discard();
	};
	let jump_taken = if target.value <= 0 && (!is_zero || unconditional)
	{
		if let Some(target_addr) = state.address.checked_sub((target.value * -2) as usize)
		{
			expect_jump(&mut expected_state, target_addr, location_addr);
			true
		}
		else
		{
			// Undefined behavior
			return TestResult::discard();
		}
	}
	else if target.value > 0 && (is_zero || unconditional)
	{
		// location offset is 0, so no need to add
		if let Some(target_addr) = state
			.address
			.checked_add(((target.value + location.value + 1) * 2) as usize)
		{
			// Discard test if jumping past trigger location
			if test_state
				.frame
				.branches
				.iter()
				.find(|(k, _)| **k < target_addr)
				.is_some()
			{
				return TestResult::discard();
			}

			expect_jump(&mut expected_state, target_addr, location_addr);
			true
		}
		else
		{
			// Undefined behavior
			return TestResult::discard();
		}
	}
	else
	{
		if may_trigger
		{
			// Branch not taken, should just advance
			expected_state.address += 2;
		}
		false
	};

	let mut expected_metrics: TrackReport = [
		(Metric::InstructionReads, 1),
		(Metric::IssuedBranches, jump_taken as usize),
	]
	.into();
	if !unconditional
	{
		expected_metrics.add_stat(Metric::ConsumedOperands, 1);
		expected_metrics.add_stat(
			Metric::ConsumedBytes,
			condition.map_or(0, |c| c.0.get_first().bytes().unwrap().len()),
		);
	}
	if may_trigger
	{
		expected_metrics.add_stat(Metric::TriggeredBranches, jump_taken as usize);
	}
	test_execution_step(
		&test_state,
		RepeatingMem::<true>(Instruction::encode(&Instruction::Jump(target, location)), 0),
		&expected_state,
		&expected_metrics,
	)
}

/// Test the immediate-jump instruction that triggers immediately
#[quickcheck]
fn jmp_imm_immediately(
	state: NoCF<ExecState>,
	target: Bits<7, true>,
	condition: ArbValue<false, false>,
) -> TestResult
{
	test_jump_immediate(
		state,
		target,
		0.try_into().unwrap(),
		Some(condition),
		|expected_state, target_addr, _| expected_state.address = target_addr,
		true,
	)
}

/// Test the immediate-jump instruction that doesn't trigger immediately
#[quickcheck]
fn jmp_imm_non_trigger(
	state: NoCF<ExecState>,
	target: Bits<7, true>,
	location: Bits<6, false>,
	condition: ArbValue<false, false>,
) -> TestResult
{
	if location.value == 0
	{
		return TestResult::discard();
	}
	test_jump_immediate(
		state,
		target,
		location,
		Some(condition),
		|expected_state, target_addr, location_addr| {
			expected_state
				.frame
				.branches
				.insert(location_addr, ControlFlowType::Branch(target_addr));
		},
		false,
	)
}

/// Test the unconditional jump instruction that triggers immediately
#[quickcheck]
fn jmp_unconditional_immediately(state: NoCF<ExecState>, target: Bits<7, true>) -> TestResult
{
	test_jump_immediate(
		state,
		target,
		0.try_into().unwrap(),
		None,
		|expected_state, target_addr, _| expected_state.address = target_addr,
		true,
	)
}

/// Test the unconditional jump instruction that doesn't trigger immediately
#[quickcheck]
fn jmp_unconditional_non_trigger(
	state: NoCF<ExecState>,
	target: Bits<7, true>,
	location: Bits<6, false>,
) -> TestResult
{
	if location.value == 0
	{
		return TestResult::discard();
	}
	test_jump_immediate(
		state,
		target,
		location,
		None,
		|expected_state, target_addr, location_addr| {
			expected_state
				.frame
				.branches
				.insert(location_addr, ControlFlowType::Branch(target_addr));
		},
		false,
	)
}

/// Test the triggering of a call that was previously issued.
#[quickcheck]
fn call_trigger(
	state: NoCF<ExecState>,
	instr: SupportedInstruction,
	call_target: InstrAddr,
) -> TestResult
{
	call_trigger_impl(state, instr, call_target)
}

/// Tests the triggering of a call when the preceding stack frame is empty
#[test]
fn call_trigger_empty_stack()
{
	assert!(!call_trigger_impl(
		NoCF(ExecState {
			address: 0,
			frame: Default::default(),
			frame_stack: vec![],
			stack_buffer: 0,
		}),
		SupportedInstruction(Instruction::NoOp),
		InstrAddr(0x1000)
	)
	.is_failure())
}

fn call_trigger_impl(
	NoCF(state): NoCF<ExecState>,
	instr: SupportedInstruction,
	call_target: InstrAddr,
) -> TestResult
{
	// Skip any test with immediate control flow triggering
	// or where the instruction is a call
	match instr.0
	{
		Instruction::Call(CallVariant::Call, _) => return TestResult::discard(),
		Instruction::Call(_, offset) | Instruction::Jump(_, offset) if offset.value == 0 =>
		{
			return TestResult::discard()
		},
		_ => (),
	}
	let instr_encoded = Instruction::encode(&instr.0);

	// Execute one step on the frame to get the expected result of the next
	// instruction
	let mut expected_metrics = TrackReport::new();
	let mut expected_state =
		match Executor::from_state(&state, RepeatingMem::<true>(instr_encoded, 0))
			.step(&mut expected_metrics)
		{
			Ok(exec) => exec.state(),
			_ => return TestResult::discard(),
		};
	// Add new frame with the ready list being the same as the old ready list
	let ready_list = expected_state.frame.op_queue.remove(&0);
	expected_state.frame.op_queue = advance_queue(expected_state.frame.op_queue);
	let reads = expected_state.frame.reads.clone();
	// Create expected stack frame
	let block = Block {
		address: expected_state.frame.stack.block.address + expected_state.frame.stack.base_size,
		size: expected_state.frame.stack.block.size - expected_state.frame.stack.base_size,
	};
	let frame_block = StackFrame {
		base_size: block.size,
		block,
	};
	let new_frame = CallFrameState {
		ret_addr: state.address + 2,
		branches: Default::default(),
		op_queue: HashMap::from_iter(ready_list.map(|list| (0, list)).into_iter()),
		reads,
		stack: frame_block,
	};
	expected_state.frame.stack.block.size = expected_state.frame.stack.base_size;

	expected_state
		.frame_stack
		.insert(0, std::mem::replace(&mut expected_state.frame, new_frame));
	expected_state.clean_reads();
	// Move the current address to the call target
	expected_state.address = call_target.0;

	// Add trigger metric
	expected_metrics.add_stat(Metric::TriggeredCalls, 1);

	// Construct test state
	let mut test_state = state.clone();
	test_state
		.frame
		.branches
		.insert(state.address, ControlFlowType::Call(call_target.0));

	test_execution_step(
		&test_state,
		RepeatingMem::<true>(instr_encoded, 0),
		&expected_state,
		&expected_metrics,
	)
}

/// Test executing a call instruction that immediately triggers.
#[quickcheck]
fn call_trigger_immediately(
	NoCF(state): NoCF<ExecState>,
	ArbValue(addr): ArbValue<false, false>,
) -> TestResult
{
	// The semantics of immediately triggering is the same as if
	// there already was a call issued that triggers now and the next instruction
	// does nothing.
	// Therefore, run that scenario and check that the result is the same

	let mut nop_call_state: ExecState = state.clone();
	nop_call_state.frame.branches.insert(
		state.address,
		ControlFlowType::Call(
			if let ValueType::Int(_) = addr.value_type()
			{
				if let Some(addr) = get_relative_address(state.address, addr.get_first())
				{
					addr
				}
				else
				{
					return TestResult::discard();
				}
			}
			else
			{
				get_absolute_address(addr.get_first())
			},
		),
	);
	let mut expected_metrics = TrackReport::new();
	let expected_state = match Executor::from_state(
		&nop_call_state,
		RepeatingMem::<true>(Instruction::nop().encode(), 0),
	)
	.step(&mut expected_metrics)
	{
		Ok(exec) => exec.state(),
		Err(_) => return TestResult::discard(),
	};
	expected_metrics.add_stat(IssuedCalls, 1);
	expected_metrics.add_stat(ConsumedOperands, 1);
	expected_metrics.add_stat(ConsumedBytes, addr.scale());

	let mut test_state: ExecState = state.clone();
	if let Some(list) = test_state.frame.op_queue.get_mut(&0)
	{
		list.rest.insert(
			0,
			std::mem::replace(&mut list.first, OperandState::Ready(addr)),
		);
	}
	else
	{
		test_state
			.frame
			.op_queue
			.insert(0, OperandList::new(OperandState::Ready(addr), Vec::new()));
	}

	test_execution_step(
		&test_state,
		RepeatingMem::<true>(
			Instruction::Call(CallVariant::Call, 0.try_into().unwrap()).encode(),
			0,
		),
		&expected_state,
		&expected_metrics,
	)
}

/// Test the call instruction with an absolute address operand that doesn't
/// immediately trigger.
#[quickcheck]
fn call_non_trigger(
	NoCF(state): NoCF<ExecState>,
	ArbValue(addr): ArbValue<false, false>,
	offset: Bits<6, false>,
) -> TestResult
{
	if offset.value() == 0
	{
		return TestResult::discard();
	}

	let instr_encoded = Instruction::encode(&Instruction::Call(CallVariant::Call, offset));
	let addr_value = OperandState::Ready(addr.clone());

	let mut expected_state = state.clone();
	// Call discards any operand after the first
	expected_state.frame.op_queue.remove(&0);
	expected_state.frame.clean_reads();
	expected_state.frame.op_queue = advance_queue(expected_state.frame.op_queue);
	expected_state.frame.branches.insert(
		state.address + (offset.value() as usize * 2),
		ControlFlowType::Call(
			if let ValueType::Int(_) = addr.value_type()
			{
				if let Some(addr) = get_relative_address(state.address, addr.get_first())
				{
					addr
				}
				else
				{
					return TestResult::discard();
				}
			}
			else
			{
				get_absolute_address(addr.get_first())
			},
		),
	);
	expected_state.address += 2;

	use Metric::*;
	let expected_metrics = TrackReport::from([
		(IssuedCalls, 1),
		(InstructionReads, 1),
		(ConsumedOperands, 1),
		(ConsumedBytes, addr.scale()),
	]);

	let mut test_state: ExecState = state.clone();
	if let Some(list) = test_state.frame.op_queue.get_mut(&0)
	{
		list.rest
			.insert(0, std::mem::replace(&mut list.first, addr_value));
	}
	else
	{
		test_state
			.frame
			.op_queue
			.insert(0, OperandList::new(addr_value, Vec::new()));
	}

	test_execution_step(
		&test_state,
		RepeatingMem::<true>(instr_encoded, 0),
		&expected_state,
		&expected_metrics,
	)
}
