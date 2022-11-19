use crate::{
	executor::{test_execution_step, test_metrics, ConsumingDiscarding},
	misc::{as_isize, as_usize, regress_queue, AllowWrite, RepeatingMem},
};
use quickcheck::{Arbitrary, Gen, TestResult};
use quickcheck_macros::quickcheck;
use scry_isa::{Bits, Instruction};
use scry_sim::{
	arbitrary::{NoCF, NoReads},
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
/// as a load from memory and if so, from what address.
///
/// Guarantees:
/// * Any loads will have non-Nan/Nar values
/// * No two loads will overlap in the memory they load from
/// * There is at least 1 load
#[derive(Debug, Clone)]
struct MaybeLoadValues(Vec<(Value, Option<usize>)>);
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

		// Make at least one of the values need load
		loop
		{
			let idx = usize::arbitrary(g) % values.len();
			let (old_v, addr) = values.get_mut(idx).unwrap();

			// Ensure the value isn't Nan/Nar
			let v = arb_non_nan_nar(g);

			let mut load_address = usize::arbitrary(g);
			// Ensure that the address range doesn't overflow the address space
			if load_address.checked_add(v.size()).is_none()
			{
				load_address /= 2;
			}
			// Ensure that it doesn't overlap with another load
			if taken_addresses
				.iter()
				.any(|(addr2, size2)| overlap(load_address, v.size(), *addr2, *size2))
			{
				// Try again
				continue;
			}

			*addr = Some(load_address);
			taken_addresses.push((load_address, v.size()));
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

			v.shrink().for_each(|shrunk| {
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
	NoReads(state): NoReads<ExecState>,
	// Ready list of operands to test, at least 1 of which is an operand that needs to load
	values: MaybeLoadValues,
	instruction: ConsumingDiscarding,
) -> TestResult
{
	// Ensure no values overlap with the instruction's address
	if values
		.0
		.iter()
		.any(|(v, addr)| addr.map_or(false, |addr| overlap(addr, v.size(), state.address, 2)))
	{
		return TestResult::discard();
	}

	// Create a state where all the value aren't loaded
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
			.filter(|(v, addr)| (addr % v.scale()) != 0)
			.count(),
	);
	expected_metrics.add_stat(Metric::DataReads, load_operands.clone().count());
	expected_metrics.add_stat(
		Metric::DataReadBytes,
		load_operands.clone().fold(0, |acc, (v, _)| acc + v.size()),
	);

	// Then create state version using the loads, where each value's data
	// is in memory (if it needs loading)
	let ready_or_load = |v: Value,
	                     loading: Option<usize>,
	                     read_list: &mut Vec<(usize, usize, ValueType)>,
	                     mems: &mut BlockedMemory| {
		if let Some(addr) = loading
		{
			read_list.push((addr, 1, v.value_type()));
			mems.add_block(v.get_first().bytes().unwrap().iter().cloned(), addr);
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

/// Test issuing a load with an absolute address
#[quickcheck]
fn load_issue_absolute_address(
	NoCF(state): NoCF<ExecState>,
	typ: ValueType,
	addr: Value,
	target: Bits<5, false>,
) -> TestResult
{
	// Ignore Nar/nan or signed addresses
	if addr.get_first().bytes().is_none()
	{
		return TestResult::discard();
	}
	if let ValueType::Int(_) = addr.value_type()
	{
		return TestResult::discard();
	}

	let mut test_state = state.clone();
	test_state.frame.op_queue = regress_queue(test_state.frame.op_queue);
	test_state.frame.op_queue.insert(
		0,
		OperandList::new(OperandState::Ready(addr.clone()), Vec::new()),
	);

	let mut expected_state: ExecState = state.clone();
	expected_state.address += 2;
	let read_op = OperandState::MustRead(expected_state.frame.reads.len());
	if let Some(list) = expected_state
		.frame
		.op_queue
		.get_mut(&(target.value as usize))
	{
		list.push(read_op);
	}
	else
	{
		expected_state
			.frame
			.op_queue
			.insert(target.value as usize, OperandList::new(read_op, Vec::new()));
	}
	expected_state
		.frame
		.reads
		.push((as_usize(addr.get_first()), 1, typ));
	// Because executor equality depends on the order of the read list,
	// put the expected state in an executor and extract it so that the order
	// would be the same as the test executor
	expected_state = Executor::from_state(&expected_state, RepeatingMem::<true>(0, 0)).state();

	let (signed, size) = match typ
	{
		ValueType::Int(x) => (true, x),
		ValueType::Uint(x) => (false, x),
	};

	test_execution_step(
		&test_state,
		RepeatingMem::<false>(
			Instruction::Load(signed, (size as i32).try_into().unwrap(), target).encode(),
			0,
		),
		&expected_state,
		&[
			(Metric::InstructionReads, 1),
			(Metric::QueuedReads, 1),
			(Metric::ConsumedOperands, 1),
			(Metric::ConsumedBytes, addr.size()),
		]
		.into(),
	)
}

/// Test issuing a load with an relative address
#[quickcheck]
fn load_issue_relative_address(
	NoCF(state): NoCF<ExecState>,
	typ: ValueType,
	offset: Value,
	target: Bits<5, false>,
) -> TestResult
{
	// Ignore Nar/nan or unsigned offsets
	if offset.get_first().bytes().is_none()
	{
		return TestResult::discard();
	}
	if let ValueType::Uint(_) = offset.value_type()
	{
		return TestResult::discard();
	}
	// Ignore address calculation overflow
	let offset_value = as_isize(offset.get_first());
	let abs_addr_option = if offset_value < 0
	{
		state.address.checked_sub(offset_value.abs_diff(0))
	}
	else
	{
		state.address.checked_add(offset_value.abs_diff(0))
	};
	let absolute_addr = if let Some(addr) = abs_addr_option
	{
		addr
	}
	else
	{
		return TestResult::discard();
	};

	let mut test_state = state.clone();
	test_state.frame.op_queue = regress_queue(test_state.frame.op_queue);
	test_state.frame.op_queue.insert(
		0,
		OperandList::new(OperandState::Ready(offset.clone()), Vec::new()),
	);

	let mut expected_state: ExecState = state.clone();
	expected_state.address += 2;
	let read_op = OperandState::MustRead(expected_state.frame.reads.len());
	if let Some(list) = expected_state
		.frame
		.op_queue
		.get_mut(&(target.value as usize))
	{
		list.push(read_op);
	}
	else
	{
		expected_state
			.frame
			.op_queue
			.insert(target.value as usize, OperandList::new(read_op, Vec::new()));
	}
	expected_state.frame.reads.push((absolute_addr, 1, typ));
	// Because executor equality depends on the order of the read list,
	// put the expected state in an executor and extract it so that the order
	// would be the same as the test executor
	expected_state = Executor::from_state(&expected_state, RepeatingMem::<true>(0, 0)).state();

	let (signed, size) = match typ
	{
		ValueType::Int(x) => (true, x),
		ValueType::Uint(x) => (false, x),
	};

	test_execution_step(
		&test_state,
		RepeatingMem::<false>(
			Instruction::Load(signed, (size as i32).try_into().unwrap(), target).encode(),
			0,
		),
		&expected_state,
		&[
			(Metric::InstructionReads, 1),
			(Metric::QueuedReads, 1),
			(Metric::ConsumedOperands, 1),
			(Metric::ConsumedBytes, offset.size()),
		]
		.into(),
	)
}

// Load test:
//
//
// 2. Load issue test:
//
// Simply see it go on the list
//
// 3. Nar/nar inputs
//
// 4. Invalid address
//
