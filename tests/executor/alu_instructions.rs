use byteorder::{ByteOrder, LittleEndian};
use quickcheck::TestResult;
use scry_isa::{AluVariant, Bits, Instruction};
use scryer::{
	arbitrary::{LimitedOps, NoCF, NoReads, SimpleOps},
	execution::{ExecResult, Executor},
	memory::Memory,
	ExecState, Metric, OperandState, Scalar, TrackReport, Value,
};

fn calculate_result<T>(
	sc1: &Scalar,
	sc2: &Scalar,
	semantic_fn: impl Fn(T, T) -> T,
	read: impl Fn(&[u8]) -> T,
	write: impl Fn(&mut [u8], T),
) -> Scalar
{
	let v1 = read(sc1.bytes().unwrap());
	let v2 = read(sc2.bytes().unwrap());
	let r = semantic_fn(v1, v2);
	let mut r_bytes = Vec::new();
	r_bytes.resize(sc1.bytes().unwrap().len(), 0);
	write(r_bytes.as_mut_slice(), r);
	Scalar::Val(r_bytes.into_boxed_slice())
}

#[quickcheck]
fn add(
	// RestrictedExecState(state): RestrictedExecState<false, false, 2, 2, true>,
	state: NoCF<SimpleOps<NoReads<LimitedOps<ExecState, 2, 2>>>>,
	offset: Bits<5, false>,
) -> TestResult
{
	let state = state.0 .0 .0 .0;
	state.valid();
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

	let mut nop_encoded = [0; 2];
	LittleEndian::write_u16(
		&mut nop_encoded,
		Instruction::Alu(AluVariant::Add, offset).encode(),
	);
	let exec = Executor::from_state(&state, Memory::new(nop_encoded.into(), state.address));
	let mut metrics = TrackReport::new();
	let res = match exec.step(&mut metrics)
	{
		ExecResult::Ok(exec) =>
		{
			let mut expected_state = state.clone();
			expected_state.address += 2;
			let mut new_op_q = state.frame.op_queues.clone();
			// Remove ready-queue
			let (rem1, rem2) = new_op_q
				.remove(&0)
				.map(|(op1, mut op_rest)| (op1, op_rest.pop().unwrap()))
				.unwrap();
			// Reduce all queue indexes by 1
			expected_state.frame.op_queues = new_op_q
				.into_iter()
				.map(|(idx, ops)| (idx - 1, ops))
				.collect();

			// Calculate expected result
			let v1 = rem1.extract_value();
			let v2 = rem2.extract_value();
			let mut result_scalars = Vec::new();
			let typ = v1.value_type();
			for (sc1, sc2) in v1.iter().zip(v2.iter())
			{
				use scryer::ValueType::*;
				match typ
				{
					Uint(0) =>
					{
						result_scalars.push(calculate_result(
							sc1,
							sc2,
							u8::saturating_add,
							|b| b[0],
							|b, v| b[0] = v,
						))
					},
					Uint(1) =>
					{
						result_scalars.push(calculate_result(
							sc1,
							sc2,
							u16::saturating_add,
							LittleEndian::read_u16,
							LittleEndian::write_u16,
						))
					},
					Uint(2) =>
					{
						result_scalars.push(calculate_result(
							sc1,
							sc2,
							u32::saturating_add,
							LittleEndian::read_u32,
							LittleEndian::write_u32,
						))
					},
					Uint(3) =>
					{
						result_scalars.push(calculate_result(
							sc1,
							sc2,
							u64::saturating_add,
							LittleEndian::read_u64,
							LittleEndian::write_u64,
						))
					},
					Uint(4) =>
					{
						result_scalars.push(calculate_result(
							sc1,
							sc2,
							u128::saturating_add,
							LittleEndian::read_u128,
							LittleEndian::write_u128,
						))
					},
					Int(0) =>
					{
						result_scalars.push(calculate_result(
							sc1,
							sc2,
							i8::saturating_add,
							|b| b[0] as i8,
							|b, v| b[0] = v as u8,
						))
					},
					Int(1) =>
					{
						result_scalars.push(calculate_result(
							sc1,
							sc2,
							i16::saturating_add,
							LittleEndian::read_i16,
							LittleEndian::write_i16,
						))
					},
					Int(2) =>
					{
						result_scalars.push(calculate_result(
							sc1,
							sc2,
							i32::saturating_add,
							LittleEndian::read_i32,
							LittleEndian::write_i32,
						))
					},
					Int(3) =>
					{
						result_scalars.push(calculate_result(
							sc1,
							sc2,
							i64::saturating_add,
							LittleEndian::read_i64,
							LittleEndian::write_i64,
						))
					},
					Int(4) =>
					{
						result_scalars.push(calculate_result(
							sc1,
							sc2,
							i128::saturating_add,
							LittleEndian::read_i128,
							LittleEndian::write_i128,
						))
					},
					Uint(_) | Int(_) => unreachable!(),
				}
			}
			let result_value =
				Value::new_typed(typ, result_scalars.remove(0), result_scalars).unwrap();

			if let Some((_, op_rest)) = expected_state
				.frame
				.op_queues
				.get_mut(&(offset.value as usize))
			{
				op_rest.push(OperandState::Ready(result_value));
			}
			else
			{
				expected_state.frame.op_queues.insert(
					offset.value as usize,
					(OperandState::Ready(result_value), vec![]),
				);
			}

			if exec.state() != expected_state
			{
				TestResult::error(format!("{:?} != {:?}", exec.state(), expected_state))
			}
			else
			{
				// Check metrics
				let expected_mets: TrackReport = [
					(Metric::InstructionReads, 1),
					(Metric::QueuedValues, 1),
					(Metric::QueuedValueBytes, typ.scale()),
					(Metric::ConsumedOperands, 2),
					(Metric::ConsumedBytes, typ.scale() * 2),
				]
				.into();

				if metrics != expected_mets
				{
					TestResult::error(format!("{:?} != {:?}", metrics, expected_mets))
				}
				else
				{
					TestResult::from_bool(true)
				}
			}
		},
		err => TestResult::error(format!("Unexpected: {:?}", err)),
	};
	res
}
