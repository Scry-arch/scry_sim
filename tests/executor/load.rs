use crate::{
	executor::{test_execution_step, test_metrics, ConsumingDiscarding},
	misc::{
		get_absolute_address, get_indexed_address, get_relative_address, regress_queue, AllowWrite,
		RepeatingMem,
	},
};
use quickcheck::{Arbitrary, Gen, TestResult};
use quickcheck_macros::quickcheck;
use scry_isa::Instruction;
use scry_sim::{
	arbitrary::{ArbScalarVal, ArbValue, NoCF, NoReads},
	BlockedMemory, ExecState, Executor, Metric, MetricTracker, OperandList, OperandState,
	TrackReport, Value, ValueType,
};

/// Returns whether the two address range given overlap in memory.
fn overlap(addr1: usize, length1: usize, addr2: usize, length2: usize) -> bool
{
	if addr1 == addr2
	{
		true
	}
	else if addr1 < addr2
	{
		(addr1 + length1) >= addr2
	}
	else
	{
		(addr2 + length2) >= addr1
	}
}

/// A list of at least 1 value and whether that value should be tested
/// as a load from memory and if so, whether its a stack load and from what
/// address/index.
///
/// Guarantees:
/// * Any loads will have non-Nan/Nar values
/// * No two loads will overlap in the memory they load from
/// * There is at least 1 load
#[derive(Debug, Clone)]
struct MaybeLoadValues(Vec<(Value, Option<(bool, usize)>)>);
impl Arbitrary for MaybeLoadValues
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		let arb_non_nan_nar = |g: &mut Gen| {
			let mut arb = Value::arbitrary(g);
			while arb.get_first().bytes().is_none()
			{
				arb = Value::arbitrary(g);
			}
			arb
		};
		let mut values: Vec<_> = Vec::<()>::arbitrary(g)
			.into_iter()
			.map(|_| (Value::arbitrary(g), None))
			.collect();
		// We want at least 1 value
		values.push((Value::arbitrary(g), None));

		// Track used load address ranges
		let mut taken_addresses = Vec::new();
		let mut taken_indexes = Vec::new();

		// Make at least one of the values need load
		loop
		{
			let idx = usize::arbitrary(g) % values.len();
			let (old_v, addr) = values.get_mut(idx).unwrap();

			// Ensure the value isn't Nan/Nar
			let v = arb_non_nan_nar(g);

			let is_stack = bool::arbitrary(g);

			// if stack, this is the index, else absolute address
			let mut load_address = usize::arbitrary(g);
			if is_stack
			{
				load_address %= g.size() * 1000; // Set limit to stack frame size
			}
			else if load_address.checked_add(v.size()).is_none()
			{
				// Ensure that the address range doesn't overflow the address space
				load_address /= 2;
			}

			// Ensure that it doesn't overlap with another load
			if if is_stack
			{
				&taken_indexes
			}
			else
			{
				&taken_addresses
			}
			.iter()
			.any(|(addr2, size2)| {
				overlap(
					if is_stack
					{
						load_address * v.size()
					}
					else
					{
						load_address
					},
					v.size(),
					*addr2,
					*size2,
				)
			})
			{
				// Try again
				continue;
			}

			*addr = Some((is_stack, load_address));
			if is_stack
			{
				taken_indexes.push((load_address * v.size(), v.size()));
			}
			else
			{
				taken_addresses.push((load_address, v.size()));
			}
			*old_v = v;

			if bool::arbitrary(g)
			{
				break;
			}
		}

		Self(values)
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		let mut result = Vec::new();

		// Make a load into a non-load
		let mut has_at_least_1_load = false;
		let mut clone = self.clone();
		if let Some((_, addr)) = clone.0.iter_mut().find(|(_, addr)| {
			if addr.is_some()
			{
				if !has_at_least_1_load
				{
					has_at_least_1_load = true;
				}
				else
				{
					return true;
				}
			}
			false
		})
		{
			*addr = None;
			result.push(clone);
		}

		// Remove a non-load value
		if self.0.len() > 1
		{
			if let Some((idx, _)) = self
				.0
				.iter()
				.enumerate()
				.find(|(_, (_, addr))| addr.is_none())
			{
				let mut clone = self.clone();
				clone.0.remove(idx);
				result.push(clone);
			}
		}

		// Shrink values/addresses
		self.0.iter().enumerate().for_each(|(idx, (v, addr))| {
			let mut clone_idx_removed = self.clone();
			clone_idx_removed.0.remove(idx);

			v.shrink()
				.filter(|shrunk| shrunk.iter().all(|scalar| scalar.bytes().is_some()))
				.for_each(|shrunk| {
					let mut clone = clone_idx_removed.clone();
					clone.0.insert(idx, (shrunk, addr.clone()));
					result.push(clone);
				});
			addr.shrink().for_each(|shrunk| {
				let mut clone = clone_idx_removed.clone();
				clone.0.insert(idx, (v.clone(), shrunk));
				result.push(clone);
			});
		});

		Box::new(result.into_iter())
	}
}

/// Tests the triggering of a previously issued load.
#[quickcheck]
fn load_trigger(
	NoReads(mut state): NoReads<ExecState>,
	// Ready list of operands to test, at least 1 of which is an operand that needs to load
	values: MaybeLoadValues,
	instruction: ConsumingDiscarding,
) -> TestResult
{
	// Ensure no stack values overflow the address space
	let stack_block = state.frame.stack.block.clone();
	if values
		.0
		.iter()
		.filter(|(_, addr)| addr.map_or(false, |(is_stack, _)| is_stack))
		.any(|(v, addr)| {
			stack_block
				.address
				.checked_add((addr.unwrap().1 + 1) * v.size())
				.is_none()
		})
	{
		return TestResult::discard();
	}

	// Make the stack frame fit the stack reads
	state.frame.stack.block.size = values.0.iter().fold(stack_block.size, |max, (v, addr)| {
		if let Some((true, index)) = addr
		{
			std::cmp::max(max, (index + 1) * v.size())
		}
		else
		{
			max
		}
	});
	// dbg!(&state);
	// dbg!(&values);
	if
	// Ensure no non-stack read values overlap with the instruction's address or the stack
	values
		.0
		.iter()
		.filter(|(_, addr)| addr.map_or(false, |(is_stack, _)| !is_stack))
		.any(|(v, addr)| overlap(addr.unwrap().1, v.size(), state.address, 2) ||
			overlap(addr.unwrap().1, v.size(), stack_block.address, stack_block.size)
		)
		||
		// Ensure the instruction does not overlap the stack
		overlap(state.address, 2, stack_block.address, stack_block.size)
	{
		return TestResult::discard();
	}

	// Create a state where all the values aren't loaded
	let mut pretest_state = state.clone();
	// Regress the operand queue so we can add our values as the ready list
	pretest_state.frame.op_queue = regress_queue(pretest_state.frame.op_queue);
	// We add all the values as not needing loading
	pretest_state.frame.op_queue.insert(
		0,
		OperandList::new(
			OperandState::Ready(values.0[0].0.clone()),
			values.0[1..]
				.iter()
				.map(|(v, _)| OperandState::Ready(v.clone()))
				.collect(),
		),
	);

	// Execute our non-loading state to find the expected result
	let mut expected_metrics = TrackReport::new();
	let expected_result = Executor::from_state(
		&pretest_state,
		RepeatingMem::<true>(instruction.0.encode(), 0),
	)
	.step(&mut expected_metrics);

	// Add load metric to expected metrics
	let load_operands = values.0.iter().enumerate().filter_map(|(idx, (v, addr))| {
		if let Some(addr) = addr
		{
			if !instruction.discards(idx)
			{
				return Some((v, *addr));
			}
		}
		None
	});
	expected_metrics.add_stat(
		Metric::UnalignedReads,
		load_operands
			.clone()
			// Ignore stack reads (cannot be unaligned)
			.filter_map(|(v, (is_stack, addr))| if is_stack { None} else {Some((v, addr))})
			.filter(|(v, addr)| (addr % v.scale()) != 0)
			.count(),
	);
	let stack_loads = load_operands.clone().filter(|(_, (is_stack, _))| *is_stack);
	expected_metrics.add_stat(Metric::DataReads, load_operands.clone().count());
	expected_metrics.add_stat(Metric::StackReads, stack_loads.clone().count());
	expected_metrics.add_stat(
		Metric::DataReadBytes,
		load_operands.clone().fold(0, |acc, (v, _)| acc + v.size()),
	);
	expected_metrics.add_stat(
		Metric::StackReadBytes,
		stack_loads.clone().fold(0, |acc, (v, _)| acc + v.size()),
	);

	// Then create state version using the loads, where each value's data
	// is in memory (if it needs loading)
	let ready_or_load = |v: Value,
	                     loading: Option<(bool, usize)>,
	                     read_list: &mut Vec<(bool, usize, usize, ValueType)>,
	                     mems: &mut BlockedMemory| {
		if let Some((is_stack, addr)) = loading
		{
			read_list.push((is_stack, addr, 1, v.value_type()));
			if is_stack
			{
				mems.add_block(
					v.get_first().bytes().unwrap().iter().cloned(),
					// Calculate the address of the load
					((stack_block.address + v.size() - 1) & !(v.size() - 1)) + (addr * v.size()),
				);
			}
			else
			{
				mems.add_block(v.get_first().bytes().unwrap().iter().cloned(), addr);
			}
			OperandState::MustRead(read_list.len() - 1)
		}
		else
		{
			OperandState::Ready(v)
		}
	};
	let mut test_mem = BlockedMemory::new(
		instruction.0.encode().to_le_bytes().into_iter(),
		state.address,
	);
	let mut test_state: ExecState = state.clone();
	test_state.frame.op_queue = regress_queue(test_state.frame.op_queue);
	test_state.frame.op_queue.insert(
		0,
		OperandList::new(
			ready_or_load(
				values.0[0].0.clone(),
				values.0[0].1,
				&mut test_state.frame.reads,
				&mut test_mem,
			),
			values.0[1..]
				.iter()
				.map(|(v, addr)| {
					ready_or_load(
						v.clone(),
						addr.clone(),
						&mut test_state.frame.reads,
						&mut test_mem,
					)
				})
				.collect(),
		),
	);
	// dbg!(&test_mem);
	// Execute state with loads
	let mut actual_metrics = TrackReport::new();
	let exec_result =
		Executor::from_state(&test_state, AllowWrite(&mut test_mem)).step(&mut actual_metrics);

	// Check that state with loads behaves like state without loads
	let same_step = match (&exec_result, &expected_result)
	{
		(Ok(exec_result), Ok(exec_expected)) => exec_result.state() == exec_expected.state(),
		(Err(err_result), Err(err_expected)) => err_result == err_expected,
		_ => false,
	};

	if !same_step
	{
		TestResult::error(format!(
			"Unexpected execution step result (actual != expected):\n{:?}\n !=\n {:?}",
			exec_result, expected_result
		))
	}
	else
	{
		test_metrics(&expected_metrics, &actual_metrics)
	}
}

/// Tests executing load instructions.
///
/// Given the initial state, regresses the operand queue and puts the given
/// operand queue as the ready queue.
/// Then tests that when the next instruction is a load the issued load has the
/// given load type and will load from the given address. Also tests metrics.
fn test_issue_load(
	NoCF(state): NoCF<ExecState>,
	load_operands: Vec<OperandState<usize>>,
	loaded_typ: ValueType,
	is_stack: bool,
	addr: usize,
) -> TestResult
{
	let mut test_state = state.clone();
	test_state.frame.op_queue = regress_queue(test_state.frame.op_queue);
	if !load_operands.is_empty()
	{
		test_state.frame.op_queue.insert(
			0,
			OperandList::new(
				load_operands.first().unwrap().clone(),
				load_operands[1..].to_vec(),
			),
		);
	}

	let mut expected_state: ExecState = state.clone();
	expected_state.address += 2;
	let read_op = OperandState::MustRead(expected_state.frame.reads.len());
	if let Some(list) = expected_state.frame.op_queue.get_mut(&0)
	{
		list.push(read_op);
	}
	else
	{
		expected_state
			.frame
			.op_queue
			.insert(0, OperandList::new(read_op, Vec::new()));
	}
	expected_state
		.frame
		.reads
		.push((is_stack, addr, 1, loaded_typ));
	// Because executor equality depends on the order of the read list,
	// put the expected state in an executor and extract it so that the order
	// would be the same as the test executor
	expected_state = Executor::from_state(&expected_state, RepeatingMem::<true>(0, 0)).state();

	let (signed, size) = match loaded_typ
	{
		ValueType::Int(x) => (true, x),
		ValueType::Uint(x) => (false, x),
	};

	test_execution_step(
		&test_state,
		RepeatingMem::<false>(
			Instruction::Load(
				signed,
				(size as i32).try_into().unwrap(),
				if is_stack { addr as i32 } else { 255 }.try_into().unwrap(),
			)
			.encode(),
			0,
		),
		&expected_state,
		&[
			(Metric::InstructionReads, 1),
			(Metric::QueuedReads, 1),
			(Metric::ConsumedOperands, load_operands.len()),
			(
				Metric::ConsumedBytes,
				load_operands
					.iter()
					.fold(0, |sum, op| sum + op.get_value().unwrap().scale()),
			),
		]
		.into(),
	)
}

/// Test issuing a load with an absolute address
#[quickcheck]
fn load_issue_absolute_address(
	NoCF(state): NoCF<ExecState>,
	typ: ValueType,
	ArbScalarVal(addr_size_pow2, addr_scalar): ArbScalarVal,
) -> TestResult
{
	test_issue_load(
		NoCF(state),
		vec![OperandState::Ready(Value::singleton_typed(
			ValueType::Uint(addr_size_pow2),
			addr_scalar.clone(),
		))],
		typ,
		false,
		get_absolute_address(&addr_scalar),
	)
}

/// Test issuing a load with an relative address
#[quickcheck]
fn load_issue_relative_address(
	NoCF(state): NoCF<ExecState>,
	typ: ValueType,
	ArbScalarVal(offset_size_pow2, offset_scalar): ArbScalarVal,
) -> TestResult
{
	// Ignore address calculation overflow
	let absolute_addr = if let Some(addr) = get_relative_address(state.address, &offset_scalar)
	{
		addr
	}
	else
	{
		return TestResult::discard();
	};

	test_issue_load(
		NoCF(state),
		vec![OperandState::Ready(Value::singleton_typed(
			ValueType::Int(offset_size_pow2),
			offset_scalar,
		))],
		typ,
		false,
		absolute_addr,
	)
}

/// Test issuing a load with an relative address
#[quickcheck]
fn load_issue_indexed(
	NoCF(state): NoCF<ExecState>,
	loaded_typ: ValueType,
	ArbValue(base_addr): ArbValue<false, false>,
	ArbScalarVal(index_size_pow2, index_scalar): ArbScalarVal,
) -> TestResult
{
	let absolute_addr = if let Some(addr) =
		get_indexed_address(state.address, &base_addr, &index_scalar, loaded_typ.scale())
	{
		addr
	}
	else
	{
		// Ignore address calculation overflow
		return TestResult::discard();
	};

	test_issue_load(
		NoCF(state),
		vec![
			OperandState::Ready(base_addr.clone()),
			OperandState::Ready(Value::singleton_typed(
				ValueType::Uint(index_size_pow2),
				index_scalar,
			)),
		],
		loaded_typ,
		false,
		absolute_addr,
	)
}

/// Test issuing a stack load
#[quickcheck]
fn load_stack(NoCF(state): NoCF<ExecState>, loaded_typ: ValueType, idx: usize) -> TestResult
{
	test_issue_load(NoCF(state), vec![], loaded_typ, true, idx % 255)
}
