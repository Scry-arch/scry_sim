use byteorder::{ByteOrder, LittleEndian};
use quickcheck::TestResult;
use scry_isa::{AluVariant, Bits, Instruction};
use scryer::{
	arbitrary::{LimitedOps, NoCF, NoReads, SimpleOps},
	execution::{ExecResult, Executor},
	memory::Memory,
	ExecState, Metric, OperandState, Scalar, TrackReport, Value,
};

/// Manages the calculation of applying the given semantic function to the given
/// inputs.
///
/// Returns the result of the semantic function as a scalar.
///
/// `read` must be able to convert a u8 slice into the type that will be used
/// for the calculation. `write` must be able to write the result of the
/// calculation into a u8 slice.
fn calculate_result<T: Default + Copy, const OPS: usize>(
	inputs: [&Scalar; OPS],
	semantic_fn: impl Fn([T; OPS]) -> T,
	read: impl Fn(&[u8]) -> T,
	write: impl Fn(&mut [u8], T),
) -> Scalar
{
	let len = inputs[0].bytes().unwrap().len();
	let mut values = [T::default(); OPS];
	values
		.iter_mut()
		.zip(inputs.into_iter())
		.for_each(|(v, sc)| *v = read(sc.bytes().unwrap()));
	let r = semantic_fn(values);
	let mut r_bytes = Vec::new();
	r_bytes.resize(len, 0);
	write(r_bytes.as_mut_slice(), r);
	Scalar::Val(r_bytes.into_boxed_slice())
}

/// The restricted execution state we will use for generating states for all Alu
/// instruction tests.
///
/// We only test the instructions with the specific number of inputs they
/// consume and without any operands being Nan, Nar, nor needing to read from
/// memory. Since all the other cases need to be handled in the same way, they
/// will be tested in a way that is agnostic to the specific instruction variant
type AluTestState<const OPS: usize> = NoCF<SimpleOps<NoReads<LimitedOps<ExecState, OPS, OPS>>>>;

/// Tests the given Alu instruction variant on the given state with the given
/// offset in the instruction.
///
/// `OPS` must match the exact number of inputs that given instruction variant
/// consumes. `*_sem` is a semantic function for each type of integer that can
/// be used to check whether the instruction has performed correctly.
/// So they are essentially the golden model.
///
/// Tests both the resulting state after taking 1 step and that the reported
/// metrics are correct
fn test_alu_instruction<const OPS: usize>(
	state: AluTestState<OPS>,
	offset: Bits<5, false>,
	variant: AluVariant,
	u8_sem: impl Fn([u8; OPS]) -> u8,
	u16_sem: impl Fn([u16; OPS]) -> u16,
	u32_sem: impl Fn([u32; OPS]) -> u32,
	u64_sem: impl Fn([u64; OPS]) -> u64,
	u128_sem: impl Fn([u128; OPS]) -> u128,
	i8_sem: impl Fn([i8; OPS]) -> i8,
	i16_sem: impl Fn([i16; OPS]) -> i16,
	i32_sem: impl Fn([i32; OPS]) -> i32,
	i64_sem: impl Fn([i64; OPS]) -> i64,
	i128_sem: impl Fn([i128; OPS]) -> i128,
) -> TestResult
{
	let state = state.0 .0 .0 .0;

	if state
		.frame
		.op_queues
		.get(&(offset.value as usize + 1))
		.filter(|(_, op_rest)| op_rest.len() > 2)
		.is_some()
	{
		// Target queue is full, so can't push result to it
		return TestResult::discard();
	}

	// Encode instruction
	let mut encoded = [0; 2];
	LittleEndian::write_u16(&mut encoded, Instruction::Alu(variant, offset).encode());

	// Create execution from state with only access to the instruction
	let exec = Executor::from_state(&state, Memory::new(encoded.into(), state.address));
	let mut metrics = TrackReport::new();

	// Perform one step
	let res = match exec.step(&mut metrics)
	{
		ExecResult::Ok(exec) =>
		{
			// Calculate expected state after step
			let mut expected_state = state.clone();
			// Advance to next instruction
			expected_state.address += 2;

			let mut new_op_q = state.frame.op_queues.clone();
			// Remove ready-queue
			let (removed_first, removed_rest) = new_op_q.remove(&0).unwrap();
			let removed_rest_values: Vec<Value> = removed_rest
				.into_iter()
				.map(|op| op.extract_value())
				.collect();

			// Reduce all operand queue indices by 1
			expected_state.frame.op_queues = new_op_q
				.into_iter()
				.map(|(idx, ops)| (idx - 1, ops))
				.collect();

			// Calculate expected result operand
			let mut result_scalars = Vec::new();
			let first_value = removed_first.extract_value();
			let typ = first_value.value_type();
			for (scalar_idx, sc0) in first_value.iter().enumerate()
			{
				let nan = Scalar::Nan;
				let mut scalars = [&nan; OPS];
				scalars[0] = sc0;
				for input_idx in 1..OPS
				{
					scalars[input_idx] = removed_rest_values[input_idx - 1]
						.iter()
						.nth(scalar_idx)
						.unwrap();
				}

				// Calculate result based on type
				use scryer::ValueType::*;
				result_scalars.push(match typ
				{
					Uint(0) => calculate_result(scalars, &u8_sem, |b| b[0], |b, v| b[0] = v),
					Uint(1) =>
					{
						calculate_result(
							scalars,
							&u16_sem,
							LittleEndian::read_u16,
							LittleEndian::write_u16,
						)
					},
					Uint(2) =>
					{
						calculate_result(
							scalars,
							&u32_sem,
							LittleEndian::read_u32,
							LittleEndian::write_u32,
						)
					},
					Uint(3) =>
					{
						calculate_result(
							scalars,
							&u64_sem,
							LittleEndian::read_u64,
							LittleEndian::write_u64,
						)
					},
					Uint(4) =>
					{
						calculate_result(
							scalars,
							&u128_sem,
							LittleEndian::read_u128,
							LittleEndian::write_u128,
						)
					},
					Int(0) =>
					{
						calculate_result(scalars, &i8_sem, |b| b[0] as i8, |b, v| b[0] = v as u8)
					},
					Int(1) =>
					{
						calculate_result(
							scalars,
							&i16_sem,
							LittleEndian::read_i16,
							LittleEndian::write_i16,
						)
					},
					Int(2) =>
					{
						calculate_result(
							scalars,
							&i32_sem,
							LittleEndian::read_i32,
							LittleEndian::write_i32,
						)
					},
					Int(3) =>
					{
						calculate_result(
							scalars,
							&i64_sem,
							LittleEndian::read_i64,
							LittleEndian::write_i64,
						)
					},
					Int(4) =>
					{
						calculate_result(
							scalars,
							&i128_sem,
							LittleEndian::read_i128,
							LittleEndian::write_i128,
						)
					},
					Uint(_) | Int(_) => unreachable!(),
				});
			}
			let result_value =
				Value::new_typed(typ, result_scalars.remove(0), result_scalars).unwrap();

			// Put expected result in correct expected state queue
			if let Some((_, op_rest)) = expected_state
				.frame
				.op_queues
				.get_mut(&(offset.value as usize))
			{
				// Push on end of queue
				op_rest.push(OperandState::Ready(result_value));
			}
			else
			{
				// No existing queue, create it
				expected_state.frame.op_queues.insert(
					offset.value as usize,
					(OperandState::Ready(result_value), vec![]),
				);
			}

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
				// Check metrics
				let expected_mets: TrackReport = [
					(Metric::InstructionReads, 1),
					(Metric::QueuedValues, 1),
					(Metric::QueuedValueBytes, typ.scale()),
					(Metric::ConsumedOperands, OPS),
					(Metric::ConsumedBytes, typ.scale() * OPS),
				]
				.into();

				if metrics != expected_mets
				{
					TestResult::error(format!(
						"Unexpected step metrics (actual != expected):\n{:?} != {:?}",
						metrics, expected_mets
					))
				}
				else
				{
					TestResult::from_bool(true)
				}
			}
		},
		err => TestResult::error(format!("Unexpected step result: {:?}", err)),
	};
	res
}

/// Test the Alu instruction variant `Add`
#[quickcheck]
fn add(state: AluTestState<2>, offset: Bits<5, false>) -> TestResult
{
	test_alu_instruction(
		state,
		offset,
		AluVariant::Add,
		|sc| u8::saturating_add(sc[0], sc[1]),
		|sc| u16::saturating_add(sc[0], sc[1]),
		|sc| u32::saturating_add(sc[0], sc[1]),
		|sc| u64::saturating_add(sc[0], sc[1]),
		|sc| u128::saturating_add(sc[0], sc[1]),
		|sc| i8::saturating_add(sc[0], sc[1]),
		|sc| i16::saturating_add(sc[0], sc[1]),
		|sc| i32::saturating_add(sc[0], sc[1]),
		|sc| i64::saturating_add(sc[0], sc[1]),
		|sc| i128::saturating_add(sc[0], sc[1]),
	)
}

/// Test the Alu instruction variant `Inc`
#[quickcheck]
fn inc(state: AluTestState<1>, offset: Bits<5, false>) -> TestResult
{
	test_alu_instruction(
		state,
		offset,
		AluVariant::Inc,
		|sc| u8::wrapping_add(sc[0], 1),
		|sc| u16::wrapping_add(sc[0], 1),
		|sc| u32::wrapping_add(sc[0], 1),
		|sc| u64::wrapping_add(sc[0], 1),
		|sc| u128::wrapping_add(sc[0], 1),
		|sc| i8::wrapping_add(sc[0], 1),
		|sc| i16::wrapping_add(sc[0], 1),
		|sc| i32::wrapping_add(sc[0], 1),
		|sc| i64::wrapping_add(sc[0], 1),
		|sc| i128::wrapping_add(sc[0], 1),
	)
}
