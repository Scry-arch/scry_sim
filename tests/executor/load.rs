use crate::{
	executor::test_execution_step,
	misc::{get_absolute_address, get_indexed_address, get_relative_address, regress_queue},
};
use quickcheck::{Arbitrary, Gen, TestResult};
use quickcheck_macros::quickcheck;
use scry_isa::{Bits, Instruction, Type};
use scry_sim::{
	arbitrary::{ArbScalarVal, ArbValue, NoCF},
	BlockedMemory, ExecState, Metric, OperandList, Value, ValueType,
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

/// Returns the minimum stack end address to enable the given value to be store
/// at the given stack index with the given stack base.
pub fn min_stack_size(stack_base: usize, value: &Value, idx: usize) -> Option<usize>
{
	stack_base.checked_add((idx + 1) * value.size())
}

/// Returns whether the given value at the given stack index in the stack with
/// given address does not overflow the address space.
pub fn address_space_fits_stack(stack_base: usize, value: &Value, idx: usize) -> bool
{
	min_stack_size(stack_base, value, idx).is_some()
}

/// Returns the effective absolute address of the given value at the given index
/// with the given stack base.
pub fn idx_address(stack_base: usize, value: &Value, idx: usize) -> usize
{
	((stack_base + value.size() - 1) & !(value.size() - 1)) + (idx * value.size())
}

/// Tests executing load instructions.
///
/// Given the initial state, regresses the operand queue and puts the given
/// operand queue as the ready queue.
/// Then tests that when the next instruction is a load the issued load has the
/// given load type and will load from the given address. Also tests metrics.
///
/// If `output_offset` is `None`, then this is a stack load and `addr` is the
/// encoded index.
fn test_issue_load(
	NoCF(state): NoCF<ExecState>,
	load_operands: Vec<Value>,
	loaded_val: ArbValue<false, false>,
	output_offset: Option<Bits<5, false>>,
	addr: usize,
) -> TestResult
{
	let effective_address = if output_offset.is_none()
	{
		idx_address(state.frame.stack.get_base_addres(), &loaded_val.0, addr)
	}
	else
	{
		addr
	};

	// Discard test if instruction and data overlap or overflow
	if effective_address
		.checked_add(loaded_val.0.scale())
		.is_none()
		|| overlap(state.address, 2, effective_address, loaded_val.0.scale())
	{
		return TestResult::discard();
	}

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
	let loaded_op = loaded_val.0.clone();
	let target_offset = if let Some(off) = output_offset
	{
		off.value as usize
	}
	else
	{
		0
	};
	if let Some(list) = expected_state.frame.op_queue.get_mut(&target_offset)
	{
		list.push(loaded_op);
	}
	else
	{
		expected_state
			.frame
			.op_queue
			.insert(target_offset, OperandList::new(loaded_op, Vec::new()));
	}

	let type_bits: Bits<4, false> = Into::<Type>::into(loaded_val.0.typ.clone())
		.try_into()
		.unwrap();

	let instruction = if let Some(r) = output_offset
	{
		Instruction::Load(type_bits, r)
	}
	else
	{
		Instruction::LoadStack(type_bits, (addr as i32).try_into().unwrap())
	};
	let mut test_mem = BlockedMemory::new(
		instruction.encode().to_le_bytes().into_iter(),
		state.address,
	);
	test_mem.add_block(
		loaded_val.0.get_first().bytes().unwrap().iter().cloned(),
		effective_address,
	);

	let is_stack = output_offset.is_none();
	test_execution_step(
		&test_state,
		test_mem,
		&expected_state,
		&[
			(Metric::InstructionReads, 1),
			(Metric::DataReads, 1),
			(Metric::DataReadBytes, loaded_val.0.typ.scale()),
			(Metric::StackReads, if !is_stack { 0 } else { 1 }),
			(
				Metric::StackReadBytes,
				if !is_stack
				{
					0
				}
				else
				{
					loaded_val.0.typ.scale()
				},
			),
			(Metric::QueuedValues, 1),
			(Metric::QueuedValueBytes, loaded_val.0.typ.scale()),
			(Metric::ConsumedOperands, load_operands.len()),
			(
				Metric::ConsumedBytes,
				load_operands.iter().fold(0, |sum, op| sum + op.scale()),
			),
			(
				Metric::UnalignedReads,
				((effective_address % loaded_val.0.typ.scale()) != 0) as usize,
			),
		]
		.into(),
	)
}

// Test issuing a load with an absolute address
#[quickcheck]
fn load_absolute_address(
	NoCF(state): NoCF<ExecState>,
	loaded: ArbValue<false, false>,
	ArbScalarVal(addr_size_pow2, addr_scalar): ArbScalarVal,
	output: Bits<5, false>,
) -> TestResult
{
	test_issue_load(
		NoCF(state),
		vec![Value::singleton_typed(
			ValueType::Uint(addr_size_pow2),
			addr_scalar.clone(),
		)],
		loaded,
		Some(output),
		get_absolute_address(&addr_scalar),
	)
}

// Test issuing a load with an relative address
#[quickcheck]
fn load_relative_address(
	NoCF(state): NoCF<ExecState>,
	loaded: ArbValue<false, false>,
	ArbScalarVal(offset_size_pow2, offset_scalar): ArbScalarVal,
	output: Bits<5, false>,
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
		vec![Value::singleton_typed(
			ValueType::Int(offset_size_pow2),
			offset_scalar,
		)],
		loaded,
		Some(output),
		absolute_addr,
	)
}

// Test issuing a load with an relative address
#[quickcheck]
fn load_indexed(
	NoCF(state): NoCF<ExecState>,
	loaded: ArbValue<false, false>,
	ArbValue(base_addr): ArbValue<false, false>,
	ArbScalarVal(index_size_pow2, index_scalar): ArbScalarVal,
	output: Bits<5, false>,
) -> TestResult
{
	let absolute_addr = if let Some(addr) = get_indexed_address(
		state.address,
		&base_addr,
		&index_scalar,
		loaded.0.typ.scale(),
	)
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
			base_addr.clone(),
			Value::singleton_typed(ValueType::Uint(index_size_pow2), index_scalar),
		],
		loaded,
		Some(output),
		absolute_addr,
	)
}

/// Test issuing a stack load
#[quickcheck]
fn load_stack(
	NoCF(state): NoCF<ExecState>,
	loaded: ArbValue<false, false>,
	idx: Bits<5, false>,
) -> TestResult
{
	test_issue_load(NoCF(state), vec![], loaded, None, idx.value as usize)
}
