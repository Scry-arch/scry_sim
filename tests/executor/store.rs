use crate::{
	executor::{
		load::{address_space_fits_stack, idx_address, min_stack_size},
		test_execution_step, test_execution_step_exceptions,
	},
	misc::{
		get_absolute_address, get_indexed_address, get_relative_address, regress_queue,
		RepeatingMem,
	},
};
use byteorder::{ByteOrder, LittleEndian};
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_isa::Instruction;
use scry_sim::{
	arbitrary::{ArbScalarVal, ArbValue, NoCF},
	BlockedMemory, ExecState, Memory, Metric, OperandList, Value, ValueType,
};
use std::cmp::max;

/// Tests the store instruction.
fn test_store_instruction<const ADDR_OPS: usize>(
	// Start state
	NoCF(mut state): NoCF<ExecState>,
	// Store address operand
	address_operands: [Value; ADDR_OPS],
	// Value to store
	to_store: &Value,
	// Initial value of all non-instruction memory bytes
	init_mem_bytes: u8,
	// The absolute address that should be the result of the given address operands.
	// If stack store, this is the index
	store_address: usize,
	// If stack store
	is_stack: bool,
) -> TestResult
{
	let idx = store_address;
	// Don't allow the stored value to overflow the address space
	let store_address = if is_stack
	{
		// Make sure the stack is big enough
		if !address_space_fits_stack(state.frame.stack.block.address, to_store, idx)
		{
			return TestResult::discard();
		}
		state.frame.stack.block.size = max(
			state.frame.stack.block.size,
			min_stack_size(state.frame.stack.block.address, to_store, idx).unwrap(),
		);
		idx_address(state.frame.stack.block.address, to_store, idx)
	}
	else
	{
		store_address
	};

	if
	// Don't allow the stored value to overflow the address space
	store_address.checked_add(to_store.scale()).is_none() ||
		// don't let instruction and data memory overlap
		(state.address < (store_address + to_store.scale()) &&
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
	LittleEndian::write_u16(
		&mut encoded_bytes,
		if is_stack
		{
			Instruction::StoreStack((idx as i32).try_into().unwrap())
		}
		else
		{
			Instruction::Store
		}
		.encode(),
	);
	mem.add_block(encoded_bytes.into_iter(), state.address);

	let test_state = clone_with_front_operands(&state, to_store.clone(), address_operands.clone());

	let mut expected_state = state.clone();
	expected_state.address += 2;

	let step_result = test_execution_step::<BlockedMemory>(
		&test_state,
		&mut mem,
		&expected_state,
		&[
			(Metric::InstructionReads, 1),
			(Metric::ConsumedOperands, 1 + ADDR_OPS),
			(
				Metric::ConsumedBytes,
				to_store.scale() + address_operands.iter().fold(0, |sum, v| sum + v.scale()),
			),
			(Metric::DataWriteBytes, to_store.scale()),
			(
				Metric::UnalignedWrites,
				!((store_address % to_store.scale()) == 0) as usize,
			),
			(Metric::StackWrites, is_stack as usize),
			(Metric::StackWriteBytes, is_stack as usize * to_store.size()),
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
	test_state.frame.op_queue = regress_queue(test_state.frame.op_queue);
	let new_ready_list = OperandList::new(first, rest.into_iter().collect());
	test_state.frame.op_queue.insert(0, new_ready_list);
	test_state
}

/// Tests the store instruction when taking an unsigned address.
#[quickcheck]
fn store_absolute(
	NoCF(state): NoCF<ExecState>,
	ArbScalarVal(addr_size_pow2, addr_scalar): ArbScalarVal,
	ArbValue(to_store): ArbValue<false, false>,
	init_mem_bytes: u8,
) -> TestResult
{
	let store_address = get_absolute_address(&addr_scalar);

	test_store_instruction(
		NoCF(state),
		[Value::singleton_typed(
			ValueType::Uint(addr_size_pow2),
			addr_scalar,
		)],
		&to_store,
		init_mem_bytes,
		store_address,
		false,
	)
}

/// Tests the store instruction when taking a signed address.
#[quickcheck]
fn store_relative(
	NoCF(state): NoCF<ExecState>,
	ArbScalarVal(addr_size_pow2, addr_scalar): ArbScalarVal,
	ArbValue(to_store): ArbValue<false, false>,
	init_mem_bytes: u8,
) -> TestResult
{
	// Don't allow the final store address to overflow the address space
	let store_address = if let Some(addr) = get_relative_address(state.address, &addr_scalar)
	{
		addr
	}
	else
	{
		return TestResult::discard();
	};

	test_store_instruction(
		NoCF(state),
		[Value::singleton_typed(
			ValueType::Int(addr_size_pow2),
			addr_scalar,
		)],
		&to_store,
		init_mem_bytes,
		store_address,
		false,
	)
}

/// Tests the store instruction when taking an indexed address.
#[quickcheck]
fn store_indexed(
	NoCF(state): NoCF<ExecState>,
	ArbValue(base_addr): ArbValue<false, false>,
	ArbScalarVal(index_size_pow2, index_scalar): ArbScalarVal,
	ArbValue(to_store): ArbValue<false, false>,
	init_mem_bytes: u8,
) -> TestResult
{
	// Don't allow the final store address to overflow the address space
	let store_address = if let Some(addr) = get_indexed_address(
		state.address,
		&base_addr,
		&index_scalar,
		to_store.value_type().scale(),
	)
	{
		addr
	}
	else
	{
		return TestResult::discard();
	};

	test_store_instruction(
		NoCF(state),
		[
			base_addr,
			Value::singleton_typed(ValueType::Uint(index_size_pow2), index_scalar),
		],
		&to_store,
		init_mem_bytes,
		store_address,
		false,
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

	let mut expected_state = state.clone();
	expected_state.address += 2;

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

	let mut expected_state = state.clone();
	expected_state.address += 2;

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
	ArbValue(to_store): ArbValue<true, false>,
) -> TestResult
{
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
	ArbValue(address): ArbValue<true, false>,
	payload: usize,
	to_store: ValueType,
) -> TestResult
{
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

/// Tests the store instruction throws exception if an address isn't given (and
/// the value to store is not NaN)
#[quickcheck]
fn store_missing_addr(
	NoCF(mut state): NoCF<ExecState>,
	to_store: ArbValue<true, false>,
) -> TestResult
{
	// Regress the operand queue so that the ready list can be empty
	state.frame.op_queue = regress_queue(state.frame.op_queue);
	let test_state = clone_with_front_operands(&state, to_store.0.clone(), []);

	test_execution_step_exceptions(
		&test_state,
		RepeatingMem::<false>(Instruction::Store.encode(), 0),
		&[
			(Metric::InstructionReads, 1),
			(Metric::ConsumedOperands, 1),
			(Metric::ConsumedBytes, to_store.0.scale()),
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

/// Tests the store stack without extra operands
#[quickcheck]
fn store_stack(
	state: NoCF<ExecState>,
	ArbValue(to_store): ArbValue<false, false>,
	idx: usize,
	init_mem_bytes: u8,
) -> TestResult
{
	test_store_instruction(state, [], &to_store, init_mem_bytes, idx % 32, true)
}
