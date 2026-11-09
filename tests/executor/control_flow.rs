use crate::{
	executor::{add_operand_events_for_list, test_execution_step, SupportedInstruction},
	misc::{
		add_to_queue_idx, advance_queue, get_absolute_address, get_relative_address, regress_queue,
		OneShotTracker, RepeatingMem,
	},
};
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_isa::{BitValue, Bits, CallVariant, Instruction};
use scry_sim::{
	arbitrary::{ArbValue, InstrAddr, NoCF},
	Block, CallFrameState, ControlEvent, ControlFlowType, Event, EventTracker, ExecState, Executor,
	IdEvent, OperandEvent, OperandList, StackEvent, StackFrame, ValueType,
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
	// Operands expected to be at the end of the ready list after the return
	// trigger. They should be following any operands that were already there
	// from before the call.
	expected_ready_ops: Option<OperandList>,
	expected_events: OneShotTracker,
) -> TestResult
{
	let instr_encoded = Instruction::encode(&instr);

	// Construct the expected end state
	let mut expected_state: ExecState = state.clone();
	expected_state.address = frame.ret_addr;

	// Add operands and reads to expected state
	if let Some(expected_ops) = expected_ready_ops
	{
		add_to_queue_idx(&mut expected_state, 0, expected_ops.into_iter());
	}

	// Because the order of pending reads affects equality, but we don't
	// want to dictate the order, we insert the expected state into an Executor
	// and extract it immediately, ensuring the order of reads will follow the
	// implementation
	expected_state =
		Executor::from_state(&expected_state, RepeatingMem::<true>(0, 0), &mut ()).state();
	expected_state.frame.stack.block.size -=
		state.frame.stack.block.size - state.frame.stack.private_size;
	expected_state.frame.stack.block.size += frame.stack.private_size;

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
		&expected_events,
	)
}

fn return_stack_events(
	events: &mut impl EventTracker,
	test_state: &ExecState,
	callee_frame: &CallFrameState,
)
{
	events.event(Event::Stack(
		StackEvent::Base,
		test_state.frame.stack.get_base_address(),
	));
	let caller_split_addr =
		test_state.frame.stack.get_base_address() + test_state.frame.stack.private_size;
	events.event(Event::Stack(StackEvent::Split, caller_split_addr));
	events.event(Event::Stack(
		StackEvent::Top,
		caller_split_addr + callee_frame.stack.private_size,
	));
}

/// Test the triggering of a return that was previously issued.
#[quickcheck]
fn return_trigger(
	NoCF(state): NoCF<ExecState>,
	mut frame: CallFrameState,
	instr: SupportedInstruction,
) -> TestResult
{
	// Make sure the new frame adheres to the operand limit
	frame.op_queue.iter_mut().for_each(|(_, list)| {
		while list.len() > state.op_list_depth()
		{
			list.rest.remove(list.rest.len() - 1);
		}
	});

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
	let mut expected_events = OneShotTracker::new(true);
	let expected_ready_ops = match Executor::from_state(
		&ExecState {
			frame: frame.clone(),
			frame_stack: Vec::new(),
			stack_buffer: 0,
			..state
		},
		RepeatingMem::<true>(instr_encoded, 0),
		&mut expected_events,
	)
	.step(&mut expected_events)
	{
		Ok(exec) =>
		{
			let state = exec.state();
			let ready_ops = state.frame.op_queue.get(&0).cloned();
			ready_ops
		},
		_ => return TestResult::discard(),
	};
	let id = expected_events.new_id();
	expected_events.event(Event::Id(id, IdEvent::ControlTrigger(ControlEvent::Return)));

	return_stack_events(&mut expected_events, &state, &frame);

	return_trigger_impl(
		NoCF(state),
		frame,
		instr.0,
		expected_ready_ops,
		expected_events,
	)
}

/// Test return instructions with 0-offset (return immediately)
#[quickcheck]
fn return_immediately(state: NoCF<ExecState>, mut frame: CallFrameState) -> TestResult
{
	let addr = state.0.address;
	// Make sure the new frame adheres to the operand limit
	frame.op_queue.iter_mut().for_each(|(_, list)| {
		while list.len() > state.0.op_list_depth()
		{
			list.rest.remove(list.rest.len() - 1);
		}
	});

	// The operands given to the branch location must be moved into the caller
	// frame.
	let expected_ready_ops = frame.op_queue.get(&1).cloned();

	let instr = Instruction::Call(CallVariant::Ret, 0.try_into().unwrap());

	let mut expected_events = [
		Event::Instruction(addr, instr.clone()),
		Event::Id(0, IdEvent::ControlIssue(ControlEvent::Return)),
		Event::Id(0, IdEvent::ControlTrigger(ControlEvent::Return)),
	]
	.into();

	return_stack_events(&mut expected_events, &state.0, &frame);

	add_operand_events_for_list(
		&mut expected_events,
		&frame.op_queue,
		0,
		0,
		&[OperandEvent::Discard],
	);

	return_trigger_impl(
		state,
		frame,
		instr,
		expected_ready_ops,
		expected_events.into(),
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

	let instr = Instruction::Call(CallVariant::Ret, offset);
	let instr_encoded = Instruction::encode(&instr);

	let mut expected_state = state.clone();
	// Return discards any operands its given
	expected_state.frame.op_queue.remove(&0);
	expected_state.frame.op_queue = advance_queue(expected_state.frame.op_queue);
	expected_state.frame.branches.insert(
		state.address + (offset.value() as usize * 2),
		ControlFlowType::Return,
	);
	expected_state.address += 2;

	let mut expected_events = [
		Event::Instruction(state.address, instr.clone()),
		Event::Id(0, IdEvent::ControlIssue(ControlEvent::Return)),
	]
	.into();

	add_operand_events_for_list(
		&mut expected_events,
		&state.frame.op_queue,
		0,
		0,
		&[OperandEvent::Discard],
	);

	test_execution_step(
		&state,
		RepeatingMem::<true>(instr_encoded, 0),
		&expected_state,
		&expected_events,
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
	// Skip any test with immediate control flow triggering or trigger before
	// the target address or where the instruction is a branch
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
	let mut expected_events = OneShotTracker::new(true);
	let mut expected_state = match Executor::from_state(
		&state,
		RepeatingMem::<true>(instr_encoded, 0),
		&mut expected_events,
	)
	.step(&mut expected_events)
	{
		Ok(exec) => exec.state(),
		_ => return TestResult::discard(),
	};
	let id = expected_events.new_id();
	expected_events.event(Event::Id(id, IdEvent::ControlTrigger(ControlEvent::Jump)));

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
		&expected_events,
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

	// Construct trigger address earlier than the target but after current
	// address
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
		Executor::from_state(&test_state, RepeatingMem::<true>(instr_encoded, 0), &mut ())
			.step(&mut ())
			.is_err(),
	)
}

/// Test a jump instruction, regardless of variant.
///
/// Takes the given start state and the already-resolved pieces of the jump:
///
/// * `instr`: the jump instruction to execute (already carrying its
///   immediates).
/// * `ready_ops`: the operands to place on the ready list. `None` means an
///   unconditional jump (no operands). The first operand is the condition and
///   the second, if present, is the target.
/// * `target`: the effective target value, whether immediate or operand.
/// * `location`: the effective location offset, whether 6 or 13 bits.
///
/// Tests that the state almost doesn't change except the following:
///
/// * If the jump is taken, `expect_jump` should alter the expected state given
///   to it using the target and location addresses given.
/// * If `may_trigger` is true, the expected state's address is only incremented
///   by 2 if the jump is not taken, otherwise it is always incremented.
fn test_jump(
	NoCF(state): NoCF<ExecState>,
	instr: Instruction,
	ready_ops: Option<OperandList>,
	target: i128,
	location: u32,
	expect_jump: impl FnOnce(&mut ExecState, usize, usize),
	may_trigger: bool,
) -> TestResult
{
	// Construct test state
	let mut test_state: ExecState = state.clone();
	// Regress operand queue so that we can put our input as the
	// ready list.
	test_state.frame.op_queue = regress_queue(test_state.frame.op_queue);
	if let Some(ops) = &ready_ops
	{
		test_state.frame.op_queue.insert(0, ops.clone());
	}

	// Expected state should be the same as the given one (since we first
	// regressed it for our test.), except the address
	let mut expected_state: ExecState = state.clone();
	if !may_trigger
	{
		expected_state.address += 2;
	}

	let unconditional = ready_ops.is_none();
	let cond_is_zero = ready_ops.as_ref().map_or(false, |ops| {
		ops.first
			.get_first()
			.bytes()
			.unwrap()
			.iter()
			.all(|b| *b == 0)
	});
	let Some(location_addr) = state.address.checked_add((location * 2) as usize)
	else
	{
		// Undefined behavior
		return TestResult::discard();
	};
	let jump_taken = if target <= 0 && (!cond_is_zero || unconditional)
	{
		if let Some(target_addr) = state.address.checked_sub((target * -2) as usize)
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
	else if target > 0 && (cond_is_zero || unconditional)
	{
		// location offset is 0, so no need to add
		if let Some(target_addr) = state
			.address
			.checked_add(((target as u32 + location + 1) * 2) as usize)
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

	let mut expected_events: OneShotTracker =
		[Event::Instruction(test_state.address, instr.clone())].into();
	let ctrl_id = expected_events.new_id();
	if jump_taken
	{
		expected_events.event(Event::Id(
			ctrl_id,
			IdEvent::ControlIssue(ControlEvent::Jump),
		));
	}

	ready_ops.map(|ops| {
		for _ in ops
		{
			let id = expected_events.new_id();
			expected_events.event(Event::Id(id, IdEvent::Operand(OperandEvent::Consume)));
		}
	});

	if may_trigger
	{
		if jump_taken
		{
			expected_events.event(Event::Id(
				ctrl_id,
				IdEvent::ControlTrigger(ControlEvent::Jump),
			));
		}
	}
	test_execution_step(
		&test_state,
		RepeatingMem::<true>(Instruction::encode(&instr), 0),
		&expected_state,
		&expected_events,
	)
}

/// Test the jump instruction when taking 1 operand (i.e. immediate target and
/// location offsets).
///
/// If no condition is given, will test the unconditional variant of the jump
/// (no operands). See [`test_jump`] for the remaining semantics.
fn test_jump_immediate(
	state: NoCF<ExecState>,
	target: Bits<7, true>,
	location: Bits<6, false>,
	condition: Option<ArbValue<false, false>>,
	expect_jump: impl FnOnce(&mut ExecState, usize, usize),
	may_trigger: bool,
) -> TestResult
{
	test_jump(
		state,
		Instruction::Jump(target, location),
		condition.map(|c| OperandList::new(c.0, vec![])),
		target.value as i128,
		location.value as u32,
		expect_jump,
		may_trigger,
	)
}

/// Test the jump instruction when taking 2 operand (i.e. immediate location
/// offset and target from second operand).
///
/// The location (full 13-bits) is split across the two immediate fields of the
/// instruction. See [`test_jump`] for the remaining semantics.
fn test_jump_long(
	state: NoCF<ExecState>,
	target: ArbValue<false, false>,
	location: Bits<13, false>,
	condition: ArbValue<false, false>,
	expect_jump: impl FnOnce(&mut ExecState, usize, usize),
	may_trigger: bool,
) -> TestResult
{
	let location_bits = location.value as u32;
	let lower_6 = location_bits & 0b11_1111;
	// The upper 7 bits go in the signed target field, so values with the top
	// bit set must be given as the negative number with the same encoding
	let upper_7 = ((location_bits >> 6) as i32)
		.try_into()
		.unwrap_or_else(|_| ((location_bits >> 6) as i32 - 128).try_into().unwrap());

	// The executor treats the target as a 32-bit value and calculates the
	// target address in 32-bit arithmetic, so only test targets where that
	// cannot overflow (for either jump direction).
	let target_value = match target
		.0
		.get_first()
		.i128_value()
		.and_then(|v| i32::try_from(v).ok())
	{
		Some(t)
			if t.checked_neg().and_then(|t| t.checked_mul(2)).is_some()
				&& t.checked_add(location_bits as i32 + 1)
					.and_then(|v| v.checked_mul(2))
					.is_some() =>
		{
			t
		},
		_ => return TestResult::discard(),
	};
	test_jump(
		state,
		Instruction::Jump(
			upper_7,
			Bits {
				value: lower_6 as i32,
			},
		),
		Some(OperandList::new(condition.0, vec![target.0.clone()])),
		target_value as i128,
		location_bits,
		expect_jump,
		may_trigger,
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

/// Test the long jump instruction that triggers immediately
#[quickcheck]
fn jmp_long_immediately(
	state: NoCF<ExecState>,
	target: ArbValue<false, false>,
	condition: ArbValue<false, false>,
) -> TestResult
{
	test_jump_long(
		state,
		target,
		0.try_into().unwrap(),
		condition,
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

/// Test the long jump instruction that doesn't trigger immediately
#[quickcheck]
fn jmp_long_non_trigger(
	state: NoCF<ExecState>,
	target: ArbValue<false, false>,
	location: Bits<13, false>,
	condition: ArbValue<false, false>,
) -> TestResult
{
	if location.value == 0
	{
		return TestResult::discard();
	}
	test_jump_long(
		state,
		target,
		location,
		condition,
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
			addr_space: 0,
			address: 0,
			frame: Default::default(),
			frame_stack: vec![],
			stack_buffer: 0,
			more_depth: 0,
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
	let mut expected_events = OneShotTracker::new(true);
	let mut expected_state = match Executor::from_state(
		&state,
		RepeatingMem::<true>(instr_encoded, 0),
		&mut expected_events,
	)
	.step(&mut expected_events)
	{
		Ok(exec) => exec.state(),
		_ => return TestResult::discard(),
	};
	// Add new frame with the ready list being the same as the old ready list
	let ready_list = expected_state.frame.op_queue.remove(&0);
	expected_state.frame.op_queue = advance_queue(expected_state.frame.op_queue);
	// Create expected stack frame
	let block = Block {
		address: expected_state.frame.stack.get_split_address(),
		size: expected_state.frame.stack.block.size - expected_state.frame.stack.private_size,
	};
	let frame_block = StackFrame {
		private_size: block.size,
		block,
	};
	let new_frame = CallFrameState {
		ret_addr: state.address + 2,
		branches: Default::default(),
		op_queue: HashMap::from_iter(ready_list.map(|list| (0, list)).into_iter()),
		stack: frame_block,
	};

	expected_state
		.frame_stack
		.insert(0, std::mem::replace(&mut expected_state.frame, new_frame));
	// Move the current address to the call target
	expected_state.address = call_target.0;

	// Add trigger event
	let id = expected_events.new_id();
	expected_events.event(Event::Id(id, IdEvent::ControlTrigger(ControlEvent::Call)));

	expected_events.event(Event::Stack(
		StackEvent::Base,
		state.frame.stack.get_split_address(),
	));
	expected_events.event(Event::Stack(
		StackEvent::Split,
		state.frame.stack.get_top_address(),
	));
	expected_events.event(Event::Stack(
		StackEvent::Top,
		state.frame.stack.get_top_address(),
	));

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
		&expected_events,
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
	// there already was a call issued that triggers now and the next
	// instruction does nothing.
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
	let mut expected_events = OneShotTracker::new(true);
	let expected_state = match Executor::from_state(
		&nop_call_state,
		RepeatingMem::<true>(Instruction::NoOp.encode(), 0),
		&mut expected_events,
	)
	.step(&mut expected_events)
	{
		Ok(exec) => exec.state(),
		Err(_) => return TestResult::discard(),
	};

	let mut test_state: ExecState = state.clone();
	if let Some(list) = test_state.frame.op_queue.get_mut(&0)
	{
		list.rest
			.insert(0, std::mem::replace(&mut list.first, addr));
	}
	else
	{
		test_state
			.frame
			.op_queue
			.insert(0, OperandList::new(addr, Vec::new()));
	}

	let instr = Instruction::Call(CallVariant::Call, 0.try_into().unwrap());

	// Change the instruction event to the call
	expected_events.events.iter_mut().for_each(|e| {
		if let Event::Instruction(_, i) = e
		{
			*i = instr.clone();
		}
	});
	// Get the trigger ID and reuse for the issue
	expected_events
		.events
		.iter_mut()
		.find_map(|e| {
			if let Event::Id(id, IdEvent::ControlTrigger(ControlEvent::Call)) = e
			{
				Some(*id)
			}
			else
			{
				None
			}
		})
		.map(|id| expected_events.event(Event::Id(id, IdEvent::ControlIssue(ControlEvent::Call))));

	// Add consumption of address event
	let id = expected_events.new_id();
	expected_events.event(Event::Id(id, IdEvent::Operand(OperandEvent::Consume)));

	test_execution_step(
		&test_state,
		RepeatingMem::<true>(instr.encode(), 0),
		&expected_state,
		&expected_events,
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

	let instr = Instruction::Call(CallVariant::Call, offset);
	let instr_encoded = Instruction::encode(&instr);
	let addr_value = addr.clone();

	let mut expected_state = state.clone();
	// Call discards any operand after the first
	expected_state.frame.op_queue.remove(&0);
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

	let mut expected_events = [
		Event::Instruction(state.address, instr.clone()),
		Event::Id(0, IdEvent::ControlIssue(ControlEvent::Call)),
		Event::Id(1, IdEvent::Operand(OperandEvent::Consume)),
	]
	.into();
	// Any operands other than the address (not in the initial state) are
	// discarded
	add_operand_events_for_list(
		&mut expected_events,
		&state.frame.op_queue,
		0,
		0,
		&[OperandEvent::Discard],
	);

	test_execution_step(
		&test_state,
		RepeatingMem::<true>(instr_encoded, 0),
		&expected_state,
		&expected_events,
	)
}
