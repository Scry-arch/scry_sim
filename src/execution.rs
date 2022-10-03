use crate::{
	control_flow::ControlFlow, data::OperandStack, memory::Memory, value::Value, ExecState,
	MetricTracker, Scalar, ValueType,
};
use byteorder::ByteOrder;
use scry_isa::{
	Alu2OutputVariant, Alu2Variant, AluVariant, BitValue, Bits, CallVariant, Instruction,
};
use std::fmt::Debug;

/// The result of performing one execution step
#[derive(Debug)]
pub enum ExecError
{
	/// The simulation triggered an exception
	Exception,

	/// The execution caused a simulation error
	Err,
}

/// Used to execute instructions.
#[derive(Debug)]
pub struct Executor<M: Memory>
{
	control: ControlFlow,
	operands: OperandStack,
	memory: M,
}
impl<M: Memory> Executor<M>
{
	/// Constructs an executor that starts executing from the given address from
	/// the given memory. The given values are the inputs to the instruction at
	/// start_addr
	pub fn new(start_addr: usize, memory: M, ready_ops: impl Iterator<Item = Value>) -> Self
	{
		Self {
			operands: OperandStack::new(ready_ops),
			control: ControlFlow::new(start_addr),
			memory,
		}
	}

	/// Constructions a new executor that is equivalent to the given execution
	/// state and uses the given memory.
	pub fn from_state(state: &ExecState, memory: M) -> Self
	{
		Self {
			operands: state.into(),
			control: state.into(),
			memory,
		}
	}

	/// Returns the execution state equivalent to this executors state.
	pub fn state(&self) -> ExecState
	{
		let mut frames = Vec::new();

		self.control.set_frame_state(&mut frames);
		assert!(frames.len() > 0);

		self.operands.set_all_frame_states(frames.iter_mut());

		ExecState {
			address: self.control.next_addr,
			frame: frames.remove(0),
			frame_stack: frames,
		}
	}

	/// Perform one execution step.
	///
	/// An execution step starts by executing an instruction and then checks
	/// whether any control flow is triggered. If so, the control flow is then
	/// executed before the step ends.
	///
	/// Updated the given metric tracker accordingly.
	pub fn step(mut self, tracker: &mut impl MetricTracker) -> Result<Self, ExecError>
	{
		let raw_instr = self
			.memory
			.read_instr(self.control.next_addr, tracker)
			.unwrap();
		let has_non_uniform_operands = |ops: &OperandStack| {
			ops.ready_peek().count() < 1
				|| ops.ready_peek().count() > 2
				|| ops.ready_peek().any(|op| op.must_read().is_some())
				|| ops.ready_peek().any(|op| {
					let first_op_type = ops
						.ready_peek()
						.next()
						.unwrap()
						.get_value()
						.unwrap()
						.value_type();
					let val = op.get_value().unwrap();
					val.value_type() != first_op_type
						|| val.iter().any(|scalar| {
							match scalar
							{
								Scalar::Nar(_) | Scalar::Nan => true,
								_ => false,
							}
						})
				})
		};
		let instr = Instruction::decode(byteorder::LittleEndian::read_u16(&raw_instr));
		{
			use Instruction::*;
			match instr
			{
				Call(CallVariant::Ret, offset) =>
				{
					self.control.ret(
						self.control.next_addr + ((offset.value() * 2) as usize),
						tracker,
					);
					// Discard everything in the ready list
					let _ = self.operands.ready_iter(&mut self.memory, tracker);
				},
				EchoLong(offset) =>
				{
					self.operands
						.reorder_ready(offset.value() as usize, tracker);
					// Discard (now empty) ready list
					let _ = self.operands.ready_iter(&mut self.memory, tracker);
				},
				Duplicate(to_next, tar1, tar2) =>
				{
					let ops: Vec<_> = self.operands.ready_peek().cloned().collect();
					if to_next
					{
						for op in ops.clone()
						{
							self.operands.push_operand(1, op, tracker);
						}
					}
					for op in ops
					{
						self.operands
							.push_operand(tar1.value as usize + 1, op, tracker);
					}
					self.operands
						.reorder_ready(tar2.value() as usize + 1, tracker);
					// Discard (now empty) ready list
					let _ = self.operands.ready_iter(&mut self.memory, tracker);
				},
				Nop =>
				{
					// Discard ready list
					let _ = self.operands.ready_iter(&mut self.memory, tracker);
				},
				Alu(variant, offset) =>
				{
					// TODO: support different types/lengths
					if has_non_uniform_operands(&self.operands)
					{
						return Err(ExecError::Exception);
					}
					self.perform_alu(variant, offset, tracker);
				},
				Alu2(variant, out, offset) =>
				{
					// TODO: support different types/lengths
					if has_non_uniform_operands(&self.operands)
					{
						return Err(ExecError::Exception);
					}
					self.perform_alu2(variant, out, offset, tracker);
				},
				Constant(bits) =>
				{
					// Create operand from immediate and add to next list
					let new_val: Value = if bits.is_signed()
					{
						(Bits::<8, true>::try_from(bits).unwrap().value as i8).into()
					}
					else
					{
						(Bits::<8, false>::try_from(bits).unwrap().value as u8).into()
					};
					self.operands.push_operand(1, new_val.into(), tracker);

					// Forward any ready operands to the next list
					self.operands.reorder_ready(1, tracker);

					// Discard (now empty) ready list
					let _ = self.operands.ready_iter(&mut self.memory, tracker);
				},
				_ => todo!(),
			}
		}
		if self.control.next_addr(&mut self.operands, tracker)
		{
			Ok(self)
		}
		else
		{
			Err(ExecError::Err)
		}
	}

	/// Executes an Alu instruction, consuming the needed inputs and putting the
	/// result in the relevant list.
	///
	/// The given variant is the Alu instruction to execute with the offset
	/// being which list the result should go to (0 means the first list after
	/// the list of the inputs has been discarded).
	fn perform_alu(
		&mut self,
		variant: AluVariant,
		offset: Bits<5, false>,
		tracker: &mut impl MetricTracker,
	)
	{
		let (typ, mut result_scalars) = {
			use AluVariant::*;

			// Extract operands
			let mut ins = self.operands.ready_iter(&mut self.memory, tracker);
			let mut result_scalars = Vec::new();
			let in1 = ins.next().unwrap();
			let typ = in1.0.value_type();
			let in1 = in1.0.iter();

			match variant
			{
				Add | Sub =>
				{
					// Variants with 2 inputs
					let in2 = ins.next().unwrap();
					let in2 = in2.0.iter();

					let func = match variant
					{
						Add => Self::alu_add_saturated,
						Sub => Self::alu_sub_saturated,
						_ => unreachable!(),
					};

					for (sc1, sc2) in in1.zip(in2)
					{
						result_scalars.push(Scalar::Val(func(sc1, sc2, typ).into_boxed_slice()));
					}
				},
				Inc | Dec =>
				{
					// Variants with 1 input
					let func = match variant
					{
						Inc => Self::alu_increment_wrapping,
						Dec => Self::alu_decrement_wrapping,
						_ => unreachable!(),
					};

					for sc1 in in1
					{
						result_scalars.push(Scalar::Val(func(sc1, typ).into_boxed_slice()));
					}
				},
				_ => unreachable!(),
			}

			(typ, result_scalars)
		};
		self.operands.push_operand(
			offset.value as usize,
			Value::new_typed(typ, result_scalars.remove(0), result_scalars)
				.unwrap()
				.into(),
			tracker,
		);
	}

	/// Executes an Alu2 instruction, consuming the needed inputs and putting
	/// the results in the relevant operand lists.
	///
	/// The given variant is the Alu2 instruction to execute with the offset
	/// being which list a result should go to (0 means the first list after
	/// the list of the inputs has been discarded) according to the output
	/// variant.
	fn perform_alu2(
		&mut self,
		variant: Alu2Variant,
		out_var: Alu2OutputVariant,
		offset: Bits<5, false>,
		tracker: &mut impl MetricTracker,
	)
	{
		let (typ, mut result_scalars_low, mut result_scalars_high) = {
			// Extract operands
			let mut ins = self.operands.ready_iter(&mut self.memory, tracker);
			let mut result_scalars_low = Vec::new();
			let mut result_scalars_high = Vec::new();
			let in1 = ins.next().unwrap();
			let typ = in1.0.value_type();
			let in1 = in1.0.iter();

			let in2 = ins.next().unwrap();
			let in2 = in2.0.iter();

			use Alu2Variant::*;
			let func = match variant
			{
				Add => Self::alu_add_overflowing,
				Sub => Self::alu_sub_overflowing,
				_ => unreachable!(),
			};

			for (sc1, sc2) in in1.zip(in2)
			{
				let (low, high) = func(sc1, sc2, typ);
				result_scalars_low.push(low);
				result_scalars_high.push(high);
			}
			(typ, result_scalars_low, result_scalars_high)
		};

		let low = Value::new_typed(typ, result_scalars_low.remove(0), result_scalars_low)
			.unwrap()
			.into();
		let high = Value::new::<u8>(result_scalars_high.remove(0), result_scalars_high)
			.unwrap()
			.into();
		let offset = offset.value as usize;

		// Propagate results according to output variant
		use Alu2OutputVariant::*;
		let (first_offset, first_op, second_op) = match out_var
		{
			High => (offset, high, None),
			Low => (offset, low, None),
			FirstLow => (offset, low, Some(high)),
			FirstHigh => (offset, high, Some(low)),
			NextHigh => (0, high, Some(low)),
			NextLow => (0, low, Some(high)),
		};
		self.operands.push_operand(first_offset, first_op, tracker);
		if let Some(op) = second_op
		{
			self.operands.push_operand(offset, op, tracker);
		}
	}

	/// Performs a byte-wise addition with carry.
	///
	/// Returns the resulting bytes and whether the final carry was set.
	///
	/// Assumes the two given scalars are valid values (not Nan or Nar) and have
	/// the same length. The result then has the same length.
	fn alu_add_carry(in1: &Scalar, in2: &Scalar) -> (Vec<u8>, bool)
	{
		let mut result_bytes = Vec::new();
		let mut carry = false;

		for (b1, b2) in in1.bytes().unwrap().iter().zip(in2.bytes().unwrap().iter())
		{
			fn carrying_add(v1: u8, v2: u8, c: bool) -> (u8, bool)
			{
				let r = (v1 as u16) + (v2 as u16) + (c as u16);
				let carry = r > u8::MAX as u16;
				(r as u8, carry)
			}
			let (r, c) = carrying_add(*b1, *b2, carry);
			result_bytes.push(r);
			carry = c;
		}

		(result_bytes, carry)
	}

	/// Performs a saturated addition on the given scalars.
	///
	/// The bytes of the inputs are assumed to have the given type.
	///
	/// Returns the resulting bytes. Assumes the two given scalars are valid
	/// values (not Nan or Nar) and have the same length. The result then has
	/// the same length.
	fn alu_add_saturated(in1: &Scalar, in2: &Scalar, typ: ValueType) -> Vec<u8>
	{
		let (mut result_bytes, carry) = Self::alu_add_carry(in1, in2);

		// If overflow, saturate
		match (typ, carry)
		{
			(ValueType::Uint(_), true) => result_bytes.iter_mut().for_each(|b| *b = u8::MAX),
			(ValueType::Int(_), _) =>
			{
				match Self::signed_add_overflow(
					*in1.bytes().unwrap().last().unwrap(),
					*in2.bytes().unwrap().last().unwrap(),
					*result_bytes.last().unwrap(),
				)
				{
					Some(false) =>
					{
						// Underflow, set to lowest negative
						result_bytes.iter_mut().for_each(|b| *b = 0);
						*result_bytes.last_mut().unwrap() = 0b10000000u8;
					},
					Some(true) =>
					{
						// Overflow, set to highest value
						result_bytes.iter_mut().for_each(|b| *b = u8::MAX);
						*result_bytes.last_mut().unwrap() = 0b01111111u8;
					},
					_ => (),
				}
			},
			_ => (),
		}
		result_bytes
	}

	/// Performs a saturated subtraction on the given scalars.
	///
	/// The bytes of the inputs are assumed to have the given type.
	///
	/// Returns the resulting bytes. Assumes the two given scalars are valid
	/// values (not Nan or Nar) and have the same length. The result then has
	/// the same length.
	fn alu_sub_saturated(in1: &Scalar, in2: &Scalar, typ: ValueType) -> Vec<u8>
	{
		let neg_in2 = Self::negate(in2.clone(), typ);

		// Now add the first to the negative
		let mut added = Self::alu_add_carry(in1, &neg_in2).0;

		// Check for overflow
		match typ
		{
			ValueType::Uint(_) =>
			{
				if Self::unsigned_subtract_underflow(in1.bytes().unwrap(), added.as_slice())
				{
					// Set to 0
					added.iter_mut().for_each(|b| *b = 0);
				}
			},
			ValueType::Int(_) =>
			{
				match Self::signed_sub_overflow(
					*in1.bytes().unwrap().last().unwrap(),
					*in2.bytes().unwrap().last().unwrap(),
					*added.last().unwrap(),
				)
				{
					Some(true) =>
					{
						// Overflow, set to max
						added.iter_mut().for_each(|b| *b = u8::MAX);
						*added.last_mut().unwrap() = 0b01111111;
					},
					Some(false) =>
					{
						// Underflow, set to min
						added.iter_mut().for_each(|b| *b = 0);
						*added.last_mut().unwrap() = 0b10000000;
					},
					_ => (),
				}
			},
		}

		added
	}

	/// Performs a wrapping addition of the given scalar and 1.
	///
	/// The bytes of the input are assumed to have the given type.
	///
	/// Returns the resulting bytes of the same length as the input.
	fn alu_increment_wrapping(in1: &Scalar, typ: ValueType) -> Vec<u8>
	{
		Self::alu_add_carry(in1, &Self::scalar_one(typ)).0
	}

	/// Performs a wrapping subtraction of the given scalar and 1.
	///
	/// The bytes of the input are assumed to have the given type.
	///
	/// Returns the resulting bytes of the same length as the input.
	fn alu_decrement_wrapping(in1: &Scalar, typ: ValueType) -> Vec<u8>
	{
		Self::alu_add_carry(in1, &Self::negate(Self::scalar_one(typ), typ)).0
	}

	/// Performs addition on the given scalars, returning the wrapping result
	/// and whether it overflowed. Both results are of the given type
	fn alu_add_overflowing(sc1: &Scalar, sc2: &Scalar, typ: ValueType) -> (Scalar, Scalar)
	{
		let mut raw_result = Self::alu_add_carry(sc1, sc2);
		if let ValueType::Int(_) = typ
		{
			// Ensure overflow bit is set when needed
			raw_result.1 = Self::signed_add_overflow(
				*sc1.bytes().unwrap().last().unwrap(),
				*sc2.bytes().unwrap().last().unwrap(),
				*raw_result.0.last().unwrap(),
			)
			.is_some();
		}
		let high = vec![raw_result.1 as u8];
		(
			Scalar::Val(raw_result.0.into_boxed_slice()),
			Scalar::Val(high.into_boxed_slice()),
		)
	}

	/// Performs subtraction on the given scalars, returning the wrapping result
	/// and whether it overflowed. Both results are of the given type
	fn alu_sub_overflowing(sc1: &Scalar, sc2: &Scalar, typ: ValueType) -> (Scalar, Scalar)
	{
		let mut raw_result = Self::alu_add_carry(sc1, &Self::negate(sc2.clone(), typ));

		raw_result.1 = match typ
		{
			ValueType::Uint(_) =>
			{
				Self::unsigned_subtract_underflow(sc1.bytes().unwrap(), raw_result.0.as_slice())
			},
			ValueType::Int(_) =>
			{
				Self::signed_sub_overflow(
					*sc1.bytes().unwrap().last().unwrap(),
					*sc2.bytes().unwrap().last().unwrap(),
					*raw_result.0.last().unwrap(),
				)
				.is_some()
			},
		};
		let high = vec![raw_result.1 as u8];

		(
			Scalar::Val(raw_result.0.into_boxed_slice()),
			Scalar::Val(high.into_boxed_slice()),
		)
	}

	/// Return whether and which type of signed integer overflow happened
	/// following an addition.
	///
	/// If no overflow, None is returned.
	/// Otherwise, returns true if overflow, false if underflow
	///
	/// The bytes are assumed to be the highest order bytes of the two inputs to
	/// a given operation and the result.
	fn signed_add_overflow(in1: u8, in2: u8, result: u8) -> Option<bool>
	{
		// If the input both have the same sign (both positive or both
		// negative), then overflow occurs if and only if the result has
		// the opposite sign.
		// Source: https://www.doc.ic.ac.uk/~eedwards/compsys/arithmetic/index.html (2022-07-14)
		let in1_neg = Self::is_negative(in1);
		let in2_neg = Self::is_negative(in2);
		let result_neg = Self::is_negative(result);

		if in1_neg && in2_neg && !result_neg
		{
			Some(false)
		}
		else if !in1_neg && !in2_neg && result_neg
		{
			Some(true)
		}
		else
		{
			None
		}
	}

	/// Return whether and which type of signed integer overflow happened
	/// following a subtract.
	///
	/// If no overflow, None is returned.
	/// Otherwise, returns true if overflow, false if underflow
	///
	/// The bytes are assumed to be the highest order bytes of the two inputs to
	/// a given operation and the result.
	fn signed_sub_overflow(in1: u8, in2: u8, result: u8) -> Option<bool>
	{
		// If the inputs have different sign , then overflow occurs if and only if the
		// result has the same sign as the second input.
		// Source: https://www.doc.ic.ac.uk/~eedwards/compsys/arithmetic/index.html (2022-07-29)
		let in1_neg = Self::is_negative(in1);
		let in2_neg = Self::is_negative(in2);
		let result_neg = Self::is_negative(result);

		if in1_neg && !in2_neg && !result_neg
		{
			Some(false)
		}
		else if !in1_neg && in2_neg && result_neg
		{
			Some(true)
		}
		else
		{
			None
		}
	}

	/// Returns whether a unsigned integer subtraction underflow happened.
	///
	/// The slices are the bytes of the first operand of the subtract (i.e. 'a'
	/// in 'a-b') and the bytes of the result.
	/// They are assumed to be the same length.
	fn unsigned_subtract_underflow(in1: &[u8], result: &[u8]) -> bool
	{
		// If the result is larger than the first input, underflow
		for (in1_byte, result_byte) in in1.iter().zip(result.iter()).rev()
		{
			if result_byte > in1_byte
			{
				return true;
			}
			else if result_byte < in1_byte
			{
				// No underflow
				return false;
			}
			// If equal, try next byte
		}
		false
	}

	/// Returns whether this byte represents a negative two's complement value
	fn is_negative(byte: u8) -> bool
	{
		byte & 0b10000000 != 0
	}

	/// Creates a scalar with value "1" matching the given type
	fn scalar_one(typ: ValueType) -> Scalar
	{
		let mut bytes = Vec::with_capacity(typ.scale());
		bytes.push(1);
		bytes.resize(typ.scale(), 0);
		Scalar::Val(bytes.into_boxed_slice())
	}

	/// Returns the negated version of the given scalar value.
	///
	/// I.e. given 1, returns -1 (everything is treated as signed).
	fn negate(mut v: Scalar, typ: ValueType) -> Scalar
	{
		// First flip all bits in the second input
		v.set_val(
			v.bytes()
				.unwrap()
				.iter()
				.map(|v| !*v)
				.collect::<Vec<_>>()
				.as_slice(),
		);

		// Now add 1 to get the negative version of the original value
		v.set_val(Self::alu_increment_wrapping(&v, typ).as_slice());
		v
	}
}
