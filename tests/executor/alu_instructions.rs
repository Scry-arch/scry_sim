use crate::{
	executor::{check_expected, test_execution_step, test_metrics},
	misc::{advance_queue, RepeatingMem},
};
use byteorder::{ByteOrder, LittleEndian};
use duplicate::{duplicate_item, substitute};
use num_traits::{ops::overflowing::OverflowingSub, PrimInt};
use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_isa::{Alu2OutputVariant, Alu2Variant, AluVariant, Bits, Instruction};
use scry_sim::{
	arbitrary::{ArbValue, LimitedOps, NoCF, SimpleOps},
	ExecState, Executor, Metric, MetricTracker, OperandList, Scalar, TrackReport, Value, ValueType,
};
use std::{
	cmp::min,
	ops::{BitAnd, BitOr, Mul, Shl, Shr},
};

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
type AluTestState<const OPS_IN: usize> = NoCF<SimpleOps<LimitedOps<ExecState, OPS_IN, OPS_IN>>>;

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
	i8_sem: impl Fn([i8; OPS_IN]) -> [Value; OPS_OUT],
	i16_sem: impl Fn([i16; OPS_IN]) -> [Value; OPS_OUT],
	i32_sem: impl Fn([i32; OPS_IN]) -> [Value; OPS_OUT],
	i64_sem: impl Fn([i64; OPS_IN]) -> [Value; OPS_OUT],
	u128_sem: Option<impl Fn([u128; OPS_IN]) -> [Value; OPS_OUT]>,
	i128_sem: Option<impl Fn([i128; OPS_IN]) -> [Value; OPS_OUT]>,
) -> TestResult
{
	let state = state.0 .0 .0;

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

	// Calculate expected state after step
	let mut expected_state = state.clone();
	// Advance to next instruction
	expected_state.address += 2;

	let mut new_op_q = state.frame.op_queue.clone();
	// Remove ready list
	let removed_list = new_op_q.remove(&0).unwrap();
	let removed_rest_values: Vec<Value> = removed_list.rest.into_iter().collect();

	expected_state.frame.op_queue = advance_queue(new_op_q);

	// Calculate expected result operand
	let mut result_scalars = Vec::new();
	let first_value = removed_list.first;
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
			Int(0) => calculate_result(scalars, &i8_sem, |b| b[0] as i8),
			Int(1) => calculate_result(scalars, &i16_sem, LittleEndian::read_i16),
			Int(2) => calculate_result(scalars, &i32_sem, LittleEndian::read_i32),
			Int(3) => calculate_result(scalars, &i64_sem, LittleEndian::read_i64),
			Uint(4) =>
			{
				if let Some(u128_sem) = &u128_sem
				{
					calculate_result(scalars, u128_sem, LittleEndian::read_u128)
				}
				else
				{
					return TestResult::discard();
				}
			},
			Int(4) =>
			{
				if let Some(i128_sem) = &i128_sem
				{
					calculate_result(scalars, i128_sem, LittleEndian::read_i128)
				}
				else
				{
					return TestResult::discard();
				}
			},
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
			op_list.push(result_value);
		}
		else
		{
			// No existing list, create it
			expected_state
				.frame
				.op_queue
				.insert(*idx, OperandList::new(result_value, vec![]));
		}
	}

	// Expected metrics
	let expected_mets: TrackReport = [
		(Metric::InstructionReads, 1),
		(Metric::QueuedValues, OPS_OUT),
		(Metric::QueuedValueBytes, result_bytes),
		(Metric::ConsumedOperands, OPS_IN),
		(Metric::ConsumedBytes, input_typ.scale() * OPS_IN),
	]
	.into();

	test_execution_step(
		&state,
		RepeatingMem::<false>(instr.encode(), 0),
		&expected_state,
		&expected_mets,
	)
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
		|x| [i8_sem(x).into()],
		|x| [i16_sem(x).into()],
		|x| [i32_sem(x).into()],
		|x| [i64_sem(x).into()],
		Some(|x| [u128_sem(x).into()]),
		Some(|x| [i128_sem(x).into()]),
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
	substitute! { [not_used []]
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
					|x| [i8_sem(x).idx.into()],
					|x| [i16_sem(x).idx.into()],
					|x| [i32_sem(x).idx.into()],
					|x| [i64_sem(x).idx.into()],
					Some(|x| [u128_sem(x).idx.into()]),
					Some(|x| [i128_sem(x).idx.into()]),
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
								[u8_sem]; [u16_sem]; [u32_sem]; [u64_sem];
								[i8_sem]; [i16_sem]; [i32_sem]; [i64_sem];
							]
							|x| {
								let res = sem_fn(x);
								[order]
							},
						)
						Some(
							|x| {
								let res = u128_sem(x);
								[order]
							}
						),
						Some(
							|x| {
								let res = i128_sem(x);
								[order]
							}
						),
					)
				},
			}
		}
	}
}

/// Tests the Alu instruction variants that have a std function doing the same
/// thing.
#[duplicate_item(
	test_name 		alu_var		std_fn				inputs	second;
	[add]			[Add]		[saturating_add]	[2]		[sc[1]];
	[inc_sat]		[Add]		[saturating_add]	[1]		[1];
	[sub]			[Sub]		[saturating_sub]	[2]		[sc[1]];
	[dec_sat]		[Sub]		[saturating_sub]	[1]		[1];
	[bit_and]		[BitAnd]	[bitand]			[2]		[sc[1]];
	[bit_or]		[BitOr]		[bitor]				[2]		[sc[1]];
)]
#[quickcheck]
fn test_name(state: AluTestState<inputs>, offset: Bits<5, false>) -> TestResult
{
	test_alu_instruction(
		state,
		offset,
		AluVariant::alu_var,
		|sc| u8::std_fn(sc[0], second),
		|sc| u16::std_fn(sc[0], second),
		|sc| u32::std_fn(sc[0], second),
		|sc| u64::std_fn(sc[0], second),
		|sc| u128::std_fn(sc[0], second),
		|sc| i8::std_fn(sc[0], second),
		|sc| i16::std_fn(sc[0], second),
		|sc| i32::std_fn(sc[0], second),
		|sc| i64::std_fn(sc[0], second),
		|sc| i128::std_fn(sc[0], second),
	)
}

/// Tests the Alu comparison instructions with either 1 or 2 inputs
#[duplicate_item(
	test_name 			alu_var			std_fn	inputs	second;
	[equal]				[Equal]			[eq]	[2]		[x[1]];
	[less_than]			[LessThan]		[lt]	[2]		[x[1]];
	[greater_than]		[GreaterThan]	[gt]	[2]		[x[1]];
	[equal_0]			[Equal]			[eq]	[1]		[0];
	[less_than_0]		[LessThan]		[lt]	[1]		[0];
	[greater_than_0]	[GreaterThan]	[gt]	[1]		[0];
)]
#[quickcheck]
fn test_name(state: AluTestState<inputs>, offset: Bits<5, false>) -> TestResult
{
	test_arithmetic_instruction(
		state,
		Instruction::Alu(AluVariant::alu_var, offset),
		[offset.value as usize],
		|x| [(u8::std_fn(&x[0], &second) as u8).into()],
		|x| [(u16::std_fn(&x[0], &second) as u8).into()],
		|x| [(u32::std_fn(&x[0], &second) as u8).into()],
		|x| [(u64::std_fn(&x[0], &second) as u8).into()],
		|x| [(i8::std_fn(&x[0], &second) as u8).into()],
		|x| [(i16::std_fn(&x[0], &second) as u8).into()],
		|x| [(i32::std_fn(&x[0], &second) as u8).into()],
		|x| [(i64::std_fn(&x[0], &second) as u8).into()],
		Some(|x: [u128; inputs]| [(u128::std_fn(&x[0], &second) as u8).into()]),
		Some(|x: [i128; inputs]| [(i128::std_fn(&x[0], &second) as u8).into()]),
	)
}

/// Test the Alu instruction variants that take one input, whose
/// second input is implicitly '1', and where there is a std function
/// that does the same for all data types
#[duplicate_item(
	test_name 	alu_var			std_fn;
	// [shl]		[ShiftLeft]		[shl];
	// [shr]		[ShiftRight]	[shr];
	[rol_once]	[RotateLeft]	[rotate_left];
	[ror_once]	[RotateRight]	[rotate_right];
)]
#[quickcheck]
fn test_name(state: AluTestState<1>, offset: Bits<5, false>) -> TestResult
{
	test_alu_instruction(
		state,
		offset,
		AluVariant::alu_var,
		|sc| u8::std_fn(sc[0], 1),
		|sc| u16::std_fn(sc[0], 1),
		|sc| u32::std_fn(sc[0], 1),
		|sc| u64::std_fn(sc[0], 1),
		|sc| u128::std_fn(sc[0], 1),
		|sc| i8::std_fn(sc[0], 1),
		|sc| i16::std_fn(sc[0], 1),
		|sc| i32::std_fn(sc[0], 1),
		|sc| i64::std_fn(sc[0], 1),
		|sc| i128::std_fn(sc[0], 1),
	)
}

fn conv<I: Copy>(inputs: [I; 2], semantic_fn: impl Fn(I, I) -> (I, bool)) -> (I, u8)
{
	let result = semantic_fn(inputs[0], inputs[1]);
	(result.0, result.1 as u8)
}

/// Test the Alu2 instruction variants that take one input, whose
/// second input is implicit, and where there is a std function
/// that does the same for all data types
#[duplicate_item(
	test_name 		alu_var		std_fn				implicit;
	[increment]		[Add]		[overflowing_add]	[1];
	[decrement]		[Sub]		[overflowing_sub]	[1];
)]
#[quickcheck]
fn test_name(
	state: AluTestState<1>,
	offset: Bits<5, false>,
	out_var: Alu2OutputVariant,
) -> TestResult
{
	test_alu2_instruction(
		state,
		offset,
		Alu2Variant::alu_var,
		out_var,
		|x| conv([x[0], implicit], u8::std_fn),
		|x| conv([x[0], implicit], u16::std_fn),
		|x| conv([x[0], implicit], u32::std_fn),
		|x| conv([x[0], implicit], u64::std_fn),
		|x| conv([x[0], implicit], u128::std_fn),
		|x| conv([x[0], implicit], i8::std_fn),
		|x| conv([x[0], implicit], i16::std_fn),
		|x| conv([x[0], implicit], i32::std_fn),
		|x| conv([x[0], implicit], i64::std_fn),
		|x| conv([x[0], implicit], i128::std_fn),
	)
}

/// Test the Alu2 instruction variant `Add`
#[duplicate_item(
	name			variant	func;
	[add_carry]		[Add]	[overflowing_add];
	[sub_carry]		[Sub]	[overflowing_sub];
)]
#[quickcheck]
fn name(state: AluTestState<2>, offset: Bits<5, false>, out_var: Alu2OutputVariant) -> TestResult
{
	test_alu2_instruction(
		state,
		offset,
		Alu2Variant::variant,
		out_var,
		|x| conv(x, u8::func),
		|x| conv(x, u16::func),
		|x| conv(x, u32::func),
		|x| conv(x, u64::func),
		|x| conv(x, u128::func),
		|x| conv(x, i8::func),
		|x| conv(x, i16::func),
		|x| conv(x, i32::func),
		|x| conv(x, i64::func),
		|x| conv(x, i128::func),
	)
}

/// Checks that the second ready operand is a shift amount of valid size
/// compared to the size of the first operand.
fn valid_shift<const OPS_IN: usize>(state: &AluTestState<OPS_IN>) -> Option<()>
{
	let state = &state.0 .0 .0;
	let ready = state.frame.op_queue.get(&0)?;
	let width = ready.first.typ.scale() * 8;
	let shift_type = ready.rest.first()?.typ;
	let shift_amount = if shift_type.is_signed_integer()
	{
		let amount = ready.rest.first()?.first.i128_value()?;
		if amount < 0
		{
			return None;
		}
		amount as u128
	}
	else
	{
		ready.rest.first()?.first.u128_value()?
	};

	if (width as u128) <= shift_amount
	{
		return None;
	}
	Some(())
}

#[duplicate_item(
	name				variant		func	inputs	second_input	validate;
	[multiply_carry]	[Multiply]	[mul]	[2]		[x[1]]			[Some(())];
	[multiply_implicit]	[Multiply]	[mul]	[1]		[_addr_size]	[Some(())];
	[shift_left]		[ShiftLeft]	[shl]	[2]		[x[1]]			[valid_shift(&state)];
	[shift_left_once]	[ShiftLeft]	[shl]	[1]		[1u8]			[Some(())];
)]
#[quickcheck]
fn name(
	state: AluTestState<inputs>,
	offset: Bits<5, false>,
	out_var: Alu2OutputVariant,
) -> TestResult
{
	let _addr_size = state.0 .0 .0.addr_space;
	if validate.is_none()
	{
		return TestResult::discard();
	}

	use Alu2OutputVariant::*;
	substitute! ( [
		closure(typ1, typ2) [
			|x| {
				paste::paste!{
					let mut result = [0u8;2];
					LittleEndian::[<write_ typ2>](&mut result, (x[0] as typ2).func(second_input as typ2));
					order((result[0] as typ1).into(), (result[1] as typ1).into(), out_var)
				}
			}
		];
		closure2(typ, typ2) [
			|x| {
				paste::paste!{
					let mut result = [0u8;size_of::<typ2>()];
					LittleEndian::[<write_ typ2>](&mut result, (x[0] as typ2).func(second_input as typ2));
					let (r1,r2) = result.split_at(result.len() / 2);
					order(LittleEndian::[<read_ typ>](r1).into(), LittleEndian::[<read_ typ>](r2).into(), out_var)
				}
			},
		]
	]
		if out_var == High || out_var == Low {
			fn order<T>(out1: T, out2: T, var: Alu2OutputVariant) -> [T; 1] {
				match var {
					Low => [out1],
					High => [out2],
					_ => unreachable!()
				}
			}
			test_arithmetic_instruction(
				state,
				Instruction::Alu2(Alu2Variant::variant, out_var, offset),
				[offset.value as usize],
				closure([u8],[u16]),
				duplicate! (
					[
						typ typ2 ;
						[u16] [u32];
						[u32] [u64];
						[u64] [u128];
					]
					closure2([typ], [typ2])
				)
				closure([i8],[i16]),
				duplicate! (
					[
						typ typ2 ;
						[i16] [i32];
						[i32] [i64];
						[i64] [i128];
					]
					closure2([typ], [typ2])
				)
				None::<fn(_) -> _>,
				None::<fn(_) -> _>,
			)
		} else {
			fn order<T>(out1: T, out2: T, var: Alu2OutputVariant) -> [T; 2] {
				match var {
					FirstLow => [out1, out2],
					FirstHigh => [out2, out1],
					NextLow => [out1, out2],
					NextHigh => [out2, out1],
					_ => unreachable!()
				}
			}
			test_arithmetic_instruction(
				state,
				Instruction::Alu2(Alu2Variant::variant, out_var, offset),
				[
					if out_var == FirstLow || out_var == FirstHigh { offset.value as usize}else {0},
					offset.value as usize
				],
				closure([u8],[u16]),
				duplicate! (
					[
						typ typ2 ;
						[u16] [u32];
						[u32] [u64];
						[u64] [u128];
					]
					closure2([typ], [typ2])
				)
				closure([i8],[i16]),
				duplicate! (
					[
						typ typ2 ;
						[i16] [i32];
						[i32] [i64];
						[i64] [i128];
					]
					closure2([typ], [typ2])
				)
				None::<fn(_) -> _>,
				None::<fn(_) -> _>,
			)
		}
	)
}

/// Returns the High result of shifting right by the amount of the second input
fn shr_high<T: PrimInt + OverflowingSub>(in1: T, in2: T) -> T
{
	let shift_amount = in2.to_usize().unwrap();
	if in2 == T::zero() || in1 == T::zero()
	{
		T::zero()
	}
	else
	{
		let shifted_out_bits = in1 & ((T::one() << shift_amount).overflowing_sub(&T::one()).0);
		shifted_out_bits << ((size_of::<T>() * 8) - shift_amount)
	}
}

/// Checks that the second ready operand is not a zero, returning None if so
fn valid_div<const OPS_IN: usize>(state: &AluTestState<OPS_IN>) -> Option<()>
{
	let state = &state.0 .0 .0;
	let ready = state.frame.op_queue.get(&0)?;
	ready.rest.first()?.first.u128_value().and_then(|v| {
		if v == 0
		{
			None
		}
		else
		{
			Some(())
		}
	})
}

/// Defines the rem_euclid function to return only unsigned remainders, since
/// Euclidean division only produce unsigned remainders.
trait ScryRem<Out>
{
	fn scry_rem(self, rhs: Self) -> Out;
}

#[duplicate_item(
	typ_in 	typ_out;
	[u8] 	[u8];
	[u16]	[u16];
	[u32]	[u32];
	[u64]	[u64];
	[i8] 	[u8];
	[i16]	[u16];
	[i32]	[u32];
	[i64]	[u64];
)]
impl ScryRem<typ_out> for typ_in
{
	fn scry_rem(self, rhs: typ_in) -> typ_out
	{
		typ_in::rem_euclid(self, rhs) as typ_out
	}
}

#[duplicate_item(
	name				variant		func_low(typ)		func_high(typ)	inputs	second_input	validate;
	[shift_right_once]	[ShiftRight][Shr::shr]			[shr_high]		[1]		[1]				[Some(())];
	[shift_right]		[ShiftRight][Shr::shr]			[shr_high]		[2]		[x[1]]			[valid_shift(&state)];
	[division]			[Division]	[typ::div_euclid]	[typ::scry_rem]	[2]		[x[1]]			[valid_div(&state)];
	[divide_by_two]		[Division]	[typ::div_euclid]	[typ::scry_rem]	[1]		[2]				[Some(())];
)]
#[quickcheck]
fn name(
	state: AluTestState<inputs>,
	offset: Bits<5, false>,
	out_var: Alu2OutputVariant,
) -> TestResult
{
	if validate.is_none()
	{
		return TestResult::discard();
	}

	use Alu2OutputVariant::*;
	substitute! ( [
		run_test(targets, order) [
			test_arithmetic_instruction(
				state,
				Instruction::Alu2(Alu2Variant::variant, out_var, offset),
				[targets],
				duplicate! (
					[
						typ;
						[u8]; [u16]; [u32]; [u64];
						[i8]; [i16]; [i32]; [i64];
					]
					|x| order(func_low([typ])(x[0], second_input as typ).into(), func_high([typ])(x[0], second_input).into()),
				)
				None::<fn(_) -> _>,
				None::<fn(_) -> _>,
			)
		];
	]
		if out_var == High || out_var == Low {
			let order = |out1, out2| match out_var {
					Low => [out1],
					High => [out2],
					_ => unreachable!()
				};
			run_test([offset.value as usize], [order])
		} else {
			let order = |out1, out2| match out_var {
					FirstLow => [out1, out2],
					FirstHigh => [out2, out1],
					NextLow => [out1, out2],
					NextHigh => [out2, out1],
					_ => unreachable!()
				};
			run_test(
				[
					if out_var == FirstLow || out_var == FirstHigh { offset.value as usize}else {0},
					offset.value as usize
				],
				[order]
			)
		}
	)
}

/// Tests instructions using effective types.
///
/// Converts the given values into their mutual effective type and checks
/// that running the given instruction produces the same result when using their
/// original types as their effective type.
fn test_effective_types(
	state: LimitedOps<NoCF<ExecState>, 0, 2>,
	v1: Value,
	v2: Value,
	instr: Instruction,
) -> TestResult
{
	if
	// Testing only different types
	v1.typ == v2.typ
	{
		return TestResult::discard();
	}

	let effective_typ = v1.typ.get_effective_type(&v2.typ);

	let v1_ext = Value::singleton_typed(
		effective_typ,
		v1.first.extend(effective_typ.scale(), &v1.typ),
	);
	let v2_ext = Value::singleton_typed(
		effective_typ,
		v2.first.extend(effective_typ.scale(), &v2.typ),
	);

	let mem = RepeatingMem::<false>(instr.encode(), instr.encode() as u8);

	// Execute instruction using effective types as baseline
	let mut expected_state_start = state.0 .0.clone();

	if let Some(q) = expected_state_start.frame.op_queue.get_mut(&0)
	{
		q.push_first(v2_ext.clone());
		q.push_first(v1_ext.clone());
	}
	else
	{
		expected_state_start
			.frame
			.op_queue
			.insert(0, OperandList::new(v1_ext, vec![v2_ext]));
	}

	let mut expected_metrics = TrackReport::new();
	let expected_result = Executor::from_state(&expected_state_start, mem)
		.step(&mut expected_metrics)
		.map(|ex| ex.state());
	expected_metrics.reset_stat(Metric::ConsumedBytes);
	expected_metrics.add_stat(Metric::ConsumedBytes, v1.typ.scale() + v2.typ.scale());

	// Run the original values and test the expected state
	let mut test_state = state.0 .0.clone();

	if let Some(q) = test_state.frame.op_queue.get_mut(&0)
	{
		q.push_first(v2);
		q.push_first(v1);
	}
	else
	{
		test_state
			.frame
			.op_queue
			.insert(0, OperandList::new(v1, vec![v2]));
	}

	let mut actual_metrics = TrackReport::new();
	let actual_result = Executor::from_state(&test_state, mem)
		.step(&mut actual_metrics)
		.map(|ex| ex.state());

	check_expected(
		actual_result,
		expected_result,
		|| test_metrics(&expected_metrics, &actual_metrics),
		"end state",
	)
}

/// Tests ALU instructions coerce types correctly.
#[quickcheck]
fn alu_type_coercions(
	state: LimitedOps<NoCF<ExecState>, 0, 2>,
	ArbValue(v1): ArbValue<false, false>,
	ArbValue(v2): ArbValue<false, false>,
	alu_op: AluVariant,
	target: Bits<5, false>,
) -> TestResult
{
	// Ignore ALU ops with only one input
	if vec![
		AluVariant::NarTo,
		AluVariant::IsNar,
		AluVariant::RotateLeft,
		AluVariant::RotateRight,
	]
	.contains(&alu_op)
	{
		return TestResult::discard();
	}

	test_effective_types(state, v1, v2, Instruction::Alu(alu_op, target))
}

/// Tests ALU2 instructions coerce types correctly.
#[quickcheck]
fn alu2_type_coercions(
	state: LimitedOps<NoCF<ExecState>, 0, 2>,
	ArbValue(v1): ArbValue<false, false>,
	ArbValue(v2): ArbValue<false, false>,
	alu_op: Alu2Variant,
	out_typ: Alu2OutputVariant,
	target: Bits<5, false>,
) -> TestResult
{
	// Ignore ALU ops that don't use effective types
	if vec![Alu2Variant::ShiftRight, Alu2Variant::ShiftLeft].contains(&alu_op)
	{
		return TestResult::discard();
	}
	test_effective_types(state, v1, v2, Instruction::Alu2(alu_op, out_typ, target))
}

/// Tests dividing by zero results in NaR
#[quickcheck]
fn divide_by_0(
	state: LimitedOps<NoCF<ExecState>, 0, 2>,
	ArbValue(v1): ArbValue<false, false>,
	out_var: Alu2OutputVariant,
	target: Bits<5, false>,
) -> TestResult
{
	let instr = Instruction::Alu2(Alu2Variant::Division, out_var, target.clone());

	let mut expected_state = state.0 .0.clone();
	expected_state.frame.op_queue.remove(&0);
	expected_state.frame.op_queue = advance_queue(expected_state.frame.op_queue);

	let out_low = Value::new_nar_typed(v1.typ, 0);
	let out_high = Value::new_nar_typed(ValueType::Uint(v1.typ.power()), 0);
	let (tar1, tar2, out_next) = match out_var
	{
		Alu2OutputVariant::FirstHigh => (out_high, Some(out_low), None),
		Alu2OutputVariant::FirstLow => (out_low, Some(out_high), None),
		Alu2OutputVariant::High => (out_high, None, None),
		Alu2OutputVariant::Low => (out_low, None, None),
		Alu2OutputVariant::NextHigh => (out_low, None, Some(out_high)),
		Alu2OutputVariant::NextLow => (out_high, None, Some(out_low)),
	};
	if let Some(out_next) = out_next.as_ref()
	{
		if let Some(q) = expected_state.frame.op_queue.get_mut(&0)
		{
			q.push(out_next.clone());
		}
		else
		{
			expected_state
				.frame
				.op_queue
				.insert(0, OperandList::new(out_next.clone(), vec![]));
		}
	}
	if let Some(q) = expected_state
		.frame
		.op_queue
		.get_mut(&(target.value as usize))
	{
		q.push(tar1);
		tar2.clone().into_iter().for_each(|t| q.push(t));
	}
	else
	{
		expected_state.frame.op_queue.insert(
			target.value as usize,
			OperandList::new(tar1, tar2.iter().cloned().collect()),
		);
	}
	expected_state.address += 2;

	let mut test_state = state.0 .0.clone();
	let mut v0_bytes = Vec::new();
	v0_bytes.resize(v1.typ.scale(), 0);
	let v0 = Value::singleton_typed(v1.typ, Scalar::Val(v0_bytes.into_boxed_slice()));
	if let Some(q) = test_state.frame.op_queue.get_mut(&0)
	{
		q.push_first(v0);
		q.push_first(v1.clone());
	}
	else
	{
		test_state
			.frame
			.op_queue
			.insert(0, OperandList::new(v1.clone(), vec![v0]));
	}

	let queued_values = 1 + tar2.iter().count() + out_next.iter().count();
	test_execution_step(
		&test_state,
		RepeatingMem::<false>(instr.encode(), 0),
		&expected_state,
		&[
			(Metric::ConsumedOperands, 2),
			(Metric::ConsumedBytes, v1.typ.scale() * 2),
			(Metric::QueuedValues, queued_values),
			(Metric::QueuedValueBytes, v1.typ.scale() * (queued_values)),
			(Metric::InstructionReads, 1),
		]
		.into(),
	)
}
