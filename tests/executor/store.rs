use crate::{
	executor::{test_execution_step, test_execution_step_exceptions},
	misc::{advance_queue, as_addr, regress_queue, RepeatingMem},
};
use byteorder::{ByteOrder, LittleEndian};
use quickcheck::TestResult;
use scry_isa::Instruction;
use scry_sim::{
	arbitrary::NoCF, BlockedMemory, ExecState, Memory, Metric, OperandList, OperandState, Scalar,
	Value, ValueType,
};

/// Creates a state that is identical to the given one except the address is
/// advanced by 1 and so is the operand queue
fn clone_and_advance(state: &ExecState) -> ExecState
{
	// Create the expected state after the step.
	// Should just have expended the ready list.
	let mut expected_state: ExecState = state.clone();
	expected_state.address += 2;
	expected_state.frame.op_queue.remove(&0);
	expected_state.frame.op_queue = advance_queue(expected_state.frame.op_queue);
	expected_state.frame.clean_reads();
	expected_state
}

/// Tests the store instruction.
fn test_store_instruction(
	// Start state
	NoCF(state): NoCF<ExecState>,
	// Store address operand
	address_value: &Value,
	// Value to store
	to_store: &Value,
	// Initial value of all non-instruction memory bytes
	init_mem_bytes: u8,
	// The absolute address that should be the result of the given address operand
	store_address: usize,
) -> TestResult
{
	if
	// Don't allow the stored value to overflow the address space
	store_address.checked_add(to_store.scale()).is_none() ||
		// don't let instruction and data memory overlap
		(state.address < store_address + to_store.scale() &&
			state.address+1 >= store_address)
	{
		return TestResult::discard();
	}

	// Initialize data memory
	let mut store_mem_vec = Vec::new();
	store_mem_vec.resize(to_store.scale(), init_mem_bytes);
	let mut mem = BlockedMemory::new(store_mem_vec.into_iter(), store_address);

	// Initialize instruction memory
	let mut encoded_bytes = [0u8; 2];
	LittleEndian::write_u16(&mut encoded_bytes, Instruction::Store.encode());
	mem.add_block(encoded_bytes.into_iter(), state.address);

	let test_state = clone_with_front_operands(&state, to_store.clone(), [address_value.clone()]);

	let expected_state = clone_and_advance(&state);

	let step_result = test_execution_step::<BlockedMemory>(
		&test_state,
		&mut mem,
		&expected_state,
		&[
			(Metric::InstructionReads, 1),
			(Metric::ConsumedOperands, 2),
			(
				Metric::ConsumedBytes,
				address_value.scale() + to_store.scale(),
			),
			(Metric::DataBytesWritten, to_store.scale()),
			(
				Metric::UnalignedWrites,
				!((store_address % to_store.scale()) == 0) as usize,
			),
		]
		.into(),
	);

	// Check the stored value was actually stored in memory
	if !step_result.is_failure()
	{
		if to_store
			.get_first()
			.bytes()
			.unwrap()
			.iter()
			.enumerate()
			.all(|(idx, byte)| *byte == mem.read_raw(store_address + idx).unwrap())
		{
			TestResult::passed()
		}
		else
		{
			TestResult::error(format!(
				"Unexpected end memory state (actual != expected): {:?} != {:?}",
				to_store.get_first().bytes().unwrap(),
				mem
			))
		}
	}
	else
	{
		step_result
	}
}

/// Clones the given state and sets the given operands as the first and rest
/// operands if the ready list
fn clone_with_front_operands<const N: usize>(
	state: &ExecState,
	first: Value,
	rest: [Value; N],
) -> ExecState
{
	// Initialize test state with the right operands in the ready list
	let mut test_state = state.clone();
	let new_ready_list = OperandList::new(
		OperandState::Ready(first),
		rest.into_iter().map(|v| OperandState::Ready(v)).collect(),
	);
	if let Some(ready_list) = test_state.frame.op_queue.get_mut(&0)
	{
		let old_list = std::mem::replace(ready_list, new_ready_list);
		ready_list.extend(old_list.into_iter());
	}
	else
	{
		test_state.frame.op_queue.insert(0, new_ready_list);
	}
	test_state
}

/// Tests the store instruction when taking an unsigned address.
#[quickcheck]
fn store_absolute(
	NoCF(state): NoCF<ExecState>,
	address: Value,
	to_store: Value,
	init_mem_bytes: u8,
) -> TestResult
{
	if address.get_first().bytes().is_none() || to_store.get_first().bytes().is_none()
	{
		// Ignore Nars or Nans
		return TestResult::discard();
	}
	if let ValueType::Int(_) = address.value_type()
	{
		// Ignore signed addresses
		return TestResult::discard();
	}

	let store_address = as_addr(address.get_first());

	test_store_instruction(
		NoCF(state),
		&address,
		&to_store,
		init_mem_bytes,
		store_address,
	)
}

/// Tests the store instruction when taking an signed address.
#[quickcheck]
fn store_relative(
	NoCF(state): NoCF<ExecState>,
	// state: ExecState,
	rel_address: Value,
	to_store: Value,
	init_mem_bytes: u8,
) -> TestResult
{
	if rel_address.get_first().bytes().is_none() || to_store.get_first().bytes().is_none()
	{
		// Ignore Nars or Nans
		return TestResult::discard();
	}
	if let ValueType::Uint(_) = rel_address.value_type()
	{
		// Ignore unsigned addresses
		return TestResult::discard();
	}

	let rel_store_address = as_addr(rel_address.get_first()) as isize;
	let rel_negative = rel_store_address < 0;
	let abs_rel = rel_store_address.abs_diff(0);

	let store_address = if rel_negative
	{
		state.address.checked_sub(abs_rel)
	}
	else
	{
		state.address.checked_add(abs_rel)
	};

	// Don't allow the final store address to overflow the address space
	if store_address.is_none()
	{
		return TestResult::discard();
	}

	let store_address = store_address.unwrap();

	test_store_instruction(
		NoCF(state),
		&rel_address,
		&to_store,
		init_mem_bytes,
		store_address,
	)
}

/// Tests the store instruction does nothing if the address is Nan
#[quickcheck]
fn store_nan_addr(
	NoCF(state): NoCF<ExecState>,
	address_type: ValueType,
	to_store: Value,
) -> TestResult
{
	let test_state = clone_with_front_operands(
		&state,
		to_store.clone(),
		[Value::new_nan_typed(address_type)],
	);

	let expected_state = clone_and_advance(&state);

	test_execution_step(
		&test_state,
		RepeatingMem::<false>(Instruction::Store.encode(), 0),
		&expected_state,
		&[
			(Metric::InstructionReads, 1),
			(Metric::ConsumedOperands, 2),
			(
				Metric::ConsumedBytes,
				address_type.scale() + to_store.scale(),
			),
		]
		.into(),
	)
}

/// Tests the store instruction does nothing if the address is Nan
#[quickcheck]
fn store_nan_value(NoCF(state): NoCF<ExecState>, address: Value, to_store: ValueType)
	-> TestResult
{
	let test_state =
		clone_with_front_operands(&state, Value::new_nan_typed(to_store), [address.clone()]);

	let expected_state = clone_and_advance(&state);

	test_execution_step(
		&test_state,
		RepeatingMem::<false>(Instruction::Store.encode(), 0),
		&expected_state,
		&[
			(Metric::InstructionReads, 1),
			(Metric::ConsumedOperands, 2),
			(Metric::ConsumedBytes, address.scale() + to_store.scale()),
		]
		.into(),
	)
}

/// Tests the store instruction does nothing if the address is Nan
#[quickcheck]
fn store_nar_addr(
	NoCF(state): NoCF<ExecState>,
	address_type: ValueType,
	payload: usize,
	to_store: Value,
) -> TestResult
{
	if let Scalar::Nan = to_store.get_first()
	{
		// Ignore when value is Nan
		return TestResult::discard();
	}

	let test_state = clone_with_front_operands(
		&state,
		to_store.clone(),
		[Value::new_nar_typed(address_type, payload)],
	);

	test_execution_step_exceptions(
		&test_state,
		RepeatingMem::<false>(Instruction::Store.encode(), 0),
		&[
			(Metric::InstructionReads, 1),
			(Metric::ConsumedOperands, 2),
			(
				Metric::ConsumedBytes,
				address_type.scale() + to_store.scale(),
			),
		]
		.into(),
	)
}

/// Tests the store instruction does nothing if the address is Nan
#[quickcheck]
fn store_nar_value(
	NoCF(state): NoCF<ExecState>,
	address: Value,
	payload: usize,
	to_store: ValueType,
) -> TestResult
{
	if let Scalar::Nan = address.get_first()
	{
		// Ignore when addr is Nan
		return TestResult::discard();
	}

	let test_state = clone_with_front_operands(
		&state,
		Value::new_nar_typed(to_store, payload),
		[address.clone()],
	);

	test_execution_step_exceptions(
		&test_state,
		RepeatingMem::<false>(Instruction::Store.encode(), 0),
		&[
			(Metric::InstructionReads, 1),
			(Metric::ConsumedOperands, 2),
			(Metric::ConsumedBytes, address.scale() + to_store.scale()),
		]
		.into(),
	)
}

/// Tests the store instruction throws exception is an address isn't given
#[quickcheck]
fn store_missing_addr(NoCF(mut state): NoCF<ExecState>, to_store: Value) -> TestResult
{
	// Regress the operand queue so that the ready list can be empty
	state.frame.op_queue = regress_queue(state.frame.op_queue);
	let test_state = clone_with_front_operands(&state, to_store.clone(), []);

	test_execution_step_exceptions(
		&test_state,
		RepeatingMem::<false>(Instruction::Store.encode(), 0),
		&[
			(Metric::InstructionReads, 1),
			(Metric::ConsumedOperands, 1),
			(Metric::ConsumedBytes, to_store.scale()),
		]
		.into(),
	)
}

/// Tests the store instruction throws an exception if no operands are given
#[quickcheck]
fn store_missing_operands(NoCF(mut state): NoCF<ExecState>) -> TestResult
{
	// Regress the operand queue so that the ready list can be empty
	state.frame.op_queue = regress_queue(state.frame.op_queue);

	test_execution_step_exceptions(
		&state,
		RepeatingMem::<false>(Instruction::Store.encode(), 0),
		&[(Metric::InstructionReads, 1)].into(),
	)
}
