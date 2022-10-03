use crate::misc::advance_queue;
use byteorder::{ByteOrder, LittleEndian};
use duplicate::duplicate;
use quickcheck::TestResult;
use scry_isa::{Alu2OutputVariant, Alu2Variant, AluVariant, Bits, Instruction};
use scry_sim::{
	arbitrary::{LimitedOps, NoCF, NoReads, SimpleOps},
	BlockedMemory, ExecState, Executor, Metric, OperandList, OperandState, Scalar, TrackReport,
	Value,
};
use std::cmp::min;

/// Manages the calculation of applying the given semantic function to the given
/// inputs.
///
/// Returns the result of the semantic function as singleton values.
///
/// `read` must be able to convert a u8 slice into the type that will be used
/// for the calculation.
fn calculate_result<T: Default + Copy, const OPS_IN: usize, const OPS_OUT: usize>(
	inputs: [&Scalar; OPS_IN],
	semantic_fn: impl Fn([T; OPS_IN]) -> [Value; OPS_OUT],
	read: impl Fn(&[u8]) -> T,
) -> [Value; OPS_OUT]
{
	let mut values = [T::default(); OPS_IN];
	values
		.iter_mut()
		.zip(inputs.into_iter())
		.for_each(|(v, sc)| *v = read(sc.bytes().unwrap()));
	semantic_fn(values)
}

/// The restricted execution state we will use for generating states for all Alu
/// instruction tests.
///
/// We only test the instructions with the specific number of inputs they
/// consume and without any operands being Nan, Nar, nor needing to read from
/// memory. Since all the other cases need to be handled in the same way, they
/// will be tested in a way that is agnostic to the specific instruction variant
type AluTestState<const OPS_IN: usize> =
	NoCF<SimpleOps<NoReads<LimitedOps<ExecState, OPS_IN, OPS_IN>>>>;

/// Tests the given arithmetic instruction on the given state.
///
/// `OPS_IN` must match the exact number of inputs that given instruction
/// variant consumes.
/// `OPS_OUT` must match the exact number of outputs that given instruction
/// variant produces. `*_sem` is a semantic function for each type of integer
/// that can be used to check whether the instruction has performed correctly.
/// So they are essentially the golden model.
/// `out_idx` indicates which operand list (by index) that the outputs are put
/// on.
///
/// Tests both the resulting state after taking 1 step and that the reported
/// metrics are correct
fn test_arithmetic_instruction<const OPS_IN: usize, const OPS_OUT: usize>(
	state: AluTestState<OPS_IN>,
	instr: Instruction,
	out_idx: [usize; OPS_OUT],
	u8_sem: impl Fn([u8; OPS_IN]) -> [Value; OPS_OUT],
	u16_sem: impl Fn([u16; OPS_IN]) -> [Value; OPS_OUT],
	u32_sem: impl Fn([u32; OPS_IN]) -> [Value; OPS_OUT],
	u64_sem: impl Fn([u64; OPS_IN]) -> [Value; OPS_OUT],
	u128_sem: impl Fn([u128; OPS_IN]) -> [Value; OPS_OUT],
	i8_sem: impl Fn([i8; OPS_IN]) -> [Value; OPS_OUT],
	i16_sem: impl Fn([i16; OPS_IN]) -> [Value; OPS_OUT],
	i32_sem: impl Fn([i32; OPS_IN]) -> [Value; OPS_OUT],
	i64_sem: impl Fn([i64; OPS_IN]) -> [Value; OPS_OUT],
	i128_sem: impl Fn([i128; OPS_IN]) -> [Value; OPS_OUT],
) -> TestResult
{
	let state = state.0 .0 .0 .0;

	if out_idx.iter().any(|idx| {
		state
		.frame
		.op_queue
		.get(&(idx + 1))
		// Ensure there is room on the list for the needed number of operands
		.filter(|op_list| op_list.rest.len() > (3-min(3, out_idx.iter().filter(|i| *i == idx).count())))
		.is_some()
	})
	{
		// Target operand list is full, so can't push results to it
		return TestResult::discard();
	}

	// Encode instruction
	let mut encoded = [0; 2];
	LittleEndian::write_u16(&mut encoded, instr.encode());

	// Create execution from state with only access to the instruction
	let exec = Executor::from_state(&state, BlockedMemory::new(encoded.into(), state.address));
	let mut metrics = TrackReport::new();

	// Perform one step
	let res = match exec.step(&mut metrics)
	{
		Ok(exec) =>
		{
			// Calculate expected state after step
			let mut expected_state = state.clone();
			// Advance to next instruction
			expected_state.address += 2;

			let mut new_op_q = state.frame.op_queue.clone();
			// Remove ready list
			let removed_list = new_op_q.remove(&0).unwrap();
			let removed_rest_values: Vec<Value> = removed_list
				.rest
				.into_iter()
				.map(|op| op.extract_value())
				.collect();

			expected_state.frame.op_queue = advance_queue(new_op_q);

			// Calculate expected result operand
			let mut result_scalars = Vec::new();
			let first_value = removed_list.first.extract_value();
			let input_typ = first_value.value_type();
			for (scalar_idx, sc0) in first_value.iter().enumerate()
			{
				let nan = Scalar::Nan;
				let mut scalars = [&nan; OPS_IN];
				scalars[0] = sc0;
				for input_idx in 1..OPS_IN
				{
					scalars[input_idx] = removed_rest_values[input_idx - 1]
						.iter()
						.nth(scalar_idx)
						.unwrap();
				}

				// Calculate result based on type
				use scry_sim::ValueType::*;
				result_scalars.push(match input_typ
				{
					Uint(0) => calculate_result(scalars, &u8_sem, |b| b[0]),
					Uint(1) => calculate_result(scalars, &u16_sem, LittleEndian::read_u16),
					Uint(2) => calculate_result(scalars, &u32_sem, LittleEndian::read_u32),
					Uint(3) => calculate_result(scalars, &u64_sem, LittleEndian::read_u64),
					Uint(4) => calculate_result(scalars, &u128_sem, LittleEndian::read_u128),
					Int(0) => calculate_result(scalars, &i8_sem, |b| b[0] as i8),
					Int(1) => calculate_result(scalars, &i16_sem, LittleEndian::read_i16),
					Int(2) => calculate_result(scalars, &i32_sem, LittleEndian::read_i32),
					Int(3) => calculate_result(scalars, &i64_sem, LittleEndian::read_i64),
					Int(4) => calculate_result(scalars, &i128_sem, LittleEndian::read_i128),
					Uint(_) | Int(_) => unreachable!(),
				});
			}
			let mut result_values = [(); OPS_OUT].map(|_| Value::new_nan_typed(input_typ));

			for i in 0..OPS_OUT
			{
				let mut scalars = result_scalars
					.iter()
					.map(|sc| sc[i].clone())
					.collect::<Vec<_>>();
				let first = scalars.remove(0);
				let rest: Vec<Scalar> = scalars
					.into_iter()
					.map(|v| v.iter().next().unwrap().clone())
					.collect();
				result_values[i] = Value::new_typed(
					first.value_type(),
					first.iter().next().unwrap().clone(),
					rest,
				)
				.unwrap();
			}

			let result_bytes = result_values.iter().fold(0, |acc, v| acc + v.scale());

			for (idx, result_value) in out_idx.iter().zip(result_values.into_iter())
			{
				// Put expected result in correct expected state list
				if let Some(op_list) = expected_state.frame.op_queue.get_mut(idx)
				{
					// Push on end of list
					op_list.push(OperandState::Ready(result_value));
				}
				else
				{
					// No existing list, create it
					expected_state.frame.op_queue.insert(
						*idx,
						OperandList::new(OperandState::Ready(result_value), vec![]),
					);
				}
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
					(Metric::QueuedValues, OPS_OUT),
					(Metric::QueuedValueBytes, result_bytes),
					(Metric::ConsumedOperands, OPS_IN),
					(Metric::ConsumedBytes, input_typ.scale() * OPS_IN),
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
	test_arithmetic_instruction(
		state,
		Instruction::Alu(variant, offset),
		[offset.value as usize],
		|x| [u8_sem(x).into()],
		|x| [u16_sem(x).into()],
		|x| [u32_sem(x).into()],
		|x| [u64_sem(x).into()],
		|x| [u128_sem(x).into()],
		|x| [i8_sem(x).into()],
		|x| [i16_sem(x).into()],
		|x| [i32_sem(x).into()],
		|x| [i64_sem(x).into()],
		|x| [i128_sem(x).into()],
	)
}

fn test_alu2_instruction<const OPS: usize>(
	state: AluTestState<OPS>,
	offset: Bits<5, false>,
	variant: Alu2Variant,
	out_var: Alu2OutputVariant,
	u8_sem: impl Fn([u8; OPS]) -> (u8, u8),
	u16_sem: impl Fn([u16; OPS]) -> (u16, u8),
	u32_sem: impl Fn([u32; OPS]) -> (u32, u8),
	u64_sem: impl Fn([u64; OPS]) -> (u64, u8),
	u128_sem: impl Fn([u128; OPS]) -> (u128, u8),
	i8_sem: impl Fn([i8; OPS]) -> (i8, u8),
	i16_sem: impl Fn([i16; OPS]) -> (i16, u8),
	i32_sem: impl Fn([i32; OPS]) -> (i32, u8),
	i64_sem: impl Fn([i64; OPS]) -> (i64, u8),
	i128_sem: impl Fn([i128; OPS]) -> (i128, u8),
) -> TestResult
{
	let instr = Instruction::Alu2(variant, out_var, offset);
	let offset = offset.value as usize;
	use scry_isa::Alu2OutputVariant::*;

	// Calculate expected outputs based on each Ali2OutputVariant
	// Wrap the match in non-duplicating duplicate! so that we can
	// call duplicate! in match arm positions
	duplicate! { [not_used []]
		match out_var {
			duplicate!{
				[var idx; [High] [1] ; [Low] [0];]
				var => test_arithmetic_instruction(
					state,
					instr,
					[offset],
					|x| [u8_sem(x).idx.into()],
					|x| [u16_sem(x).idx.into()],
					|x| [u32_sem(x).idx.into()],
					|x| [u64_sem(x).idx.into()],
					|x| [u128_sem(x).idx.into()],
					|x| [i8_sem(x).idx.into()],
					|x| [i16_sem(x).idx.into()],
					|x| [i32_sem(x).idx.into()],
					|x| [i64_sem(x).idx.into()],
					|x| [i128_sem(x).idx.into()],
				),
			}
			duplicate!{
				[
					var 		order				first_offset;
					[FirstLow] 	[res.0.into(), res.1.into()] 	[offset];
					[FirstHigh]	[res.1.into(), res.0.into()] 	[offset];
					[NextLow] 	[res.0.into(), res.1.into()] 	[0];
					[NextHigh]	[res.1.into(), res.0.into()] 	[0];
				]
				var => {
					test_arithmetic_instruction(
						state,
						instr,
						[first_offset, offset],
						duplicate!(
							[
								sem_fn;
								[u8_sem]; [u16_sem]; [u32_sem]; [u64_sem]; [u128_sem];
								[i8_sem]; [i16_sem]; [i32_sem]; [i64_sem]; [i128_sem];
							]
							|x| {
								let res = sem_fn(x);
								[order]
							},
						)
					)
				},
			}
		}
	}
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

/// Test the Alu2 instruction variant `Add`
#[quickcheck]
fn add_carry(
	state: AluTestState<2>,
	offset: Bits<5, false>,
	out_var: Alu2OutputVariant,
) -> TestResult
{
	fn conv<I: Copy>(inputs: [I; 2], semantic_fn: impl Fn(I, I) -> (I, bool)) -> (I, u8)
	{
		let result = semantic_fn(inputs[0], inputs[1]);
		(result.0, result.1 as u8)
	}
	test_alu2_instruction(
		state,
		offset,
		Alu2Variant::Add,
		out_var,
		|x| conv(x, u8::overflowing_add),
		|x| conv(x, u16::overflowing_add),
		|x| conv(x, u32::overflowing_add),
		|x| conv(x, u64::overflowing_add),
		|x| conv(x, u128::overflowing_add),
		|x| conv(x, i8::overflowing_add),
		|x| conv(x, i16::overflowing_add),
		|x| conv(x, i32::overflowing_add),
		|x| conv(x, i64::overflowing_add),
		|x| conv(x, i128::overflowing_add),
	)
}

/// Test the Alu instruction variant `Sub`
#[quickcheck]
fn sub(state: AluTestState<2>, offset: Bits<5, false>) -> TestResult
{
	test_alu_instruction(
		state,
		offset,
		AluVariant::Sub,
		|sc| u8::saturating_sub(sc[0], sc[1]),
		|sc| u16::saturating_sub(sc[0], sc[1]),
		|sc| u32::saturating_sub(sc[0], sc[1]),
		|sc| u64::saturating_sub(sc[0], sc[1]),
		|sc| u128::saturating_sub(sc[0], sc[1]),
		|sc| i8::saturating_sub(sc[0], sc[1]),
		|sc| i16::saturating_sub(sc[0], sc[1]),
		|sc| i32::saturating_sub(sc[0], sc[1]),
		|sc| i64::saturating_sub(sc[0], sc[1]),
		|sc| i128::saturating_sub(sc[0], sc[1]),
	)
}

/// Test the Alu instruction variant `Dec`
#[quickcheck]
fn dec(state: AluTestState<1>, offset: Bits<5, false>) -> TestResult
{
	test_alu_instruction(
		state,
		offset,
		AluVariant::Dec,
		|sc| u8::wrapping_sub(sc[0], 1),
		|sc| u16::wrapping_sub(sc[0], 1),
		|sc| u32::wrapping_sub(sc[0], 1),
		|sc| u64::wrapping_sub(sc[0], 1),
		|sc| u128::wrapping_sub(sc[0], 1),
		|sc| i8::wrapping_sub(sc[0], 1),
		|sc| i16::wrapping_sub(sc[0], 1),
		|sc| i32::wrapping_sub(sc[0], 1),
		|sc| i64::wrapping_sub(sc[0], 1),
		|sc| i128::wrapping_sub(sc[0], 1),
	)
}

/// Test the Alu2 instruction variant `Sub`
#[quickcheck]
fn sub_carry(
	state: AluTestState<2>,
	offset: Bits<5, false>,
	out_var: Alu2OutputVariant,
) -> TestResult
{
	fn conv<I: Copy>(inputs: [I; 2], semantic_fn: impl Fn(I, I) -> (I, bool)) -> (I, u8)
	{
		let result = semantic_fn(inputs[0], inputs[1]);
		(result.0, result.1 as u8)
	}
	test_alu2_instruction(
		state,
		offset,
		Alu2Variant::Sub,
		out_var,
		|x| conv(x, u8::overflowing_sub),
		|x| conv(x, u16::overflowing_sub),
		|x| conv(x, u32::overflowing_sub),
		|x| conv(x, u64::overflowing_sub),
		|x| conv(x, u128::overflowing_sub),
		|x| conv(x, i8::overflowing_sub),
		|x| conv(x, i16::overflowing_sub),
		|x| conv(x, i32::overflowing_sub),
		|x| conv(x, i64::overflowing_sub),
		|x| conv(x, i128::overflowing_sub),
	)
}
