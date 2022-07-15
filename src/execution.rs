use crate::{
	control_flow::ControlFlow, data::OperandQueue, memory::Memory, value::Value, ExecState,
	MetricTracker, Scalar, ValueType,
};
use byteorder::ByteOrder;
use scry_isa::{AluVariant, Bits, CallVariant, Instruction};
use std::fmt::Debug;

/// The result of performing one execution step
#[derive(Debug)]
pub enum ExecResult<I: Iterator<Item = Value> + Debug>
{
	/// The executor performed the step successfully
	Ok(Executor),

	/// The executor finished executing with the given result values and reports
	Done(I),

	/// The execution caused an error
	Err,
}

/// Used to execute instructions.
#[derive(Debug)]
pub struct Executor
{
	control: ControlFlow,
	operands: OperandQueue,
	memory: Memory,
}
impl Executor
{
	/// Constructs an executor that starts executing from the given address from
	/// the given memory. The given values are the inputs to the instruction at
	/// start_addr
	pub fn new(start_addr: usize, memory: Memory, ready_ops: impl Iterator<Item = Value>) -> Self
	{
		Self {
			operands: OperandQueue::new(ready_ops),
			control: ControlFlow::new(start_addr),
			memory,
		}
	}

	/// Constructions a new executor that is equivalent to the given execution
	/// state and uses the given memory.
	pub fn from_state(state: &ExecState, memory: Memory) -> Self
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
	pub fn step(
		mut self,
		tracker: &mut impl MetricTracker,
	) -> ExecResult<impl Iterator<Item = Value> + Debug>
	{
		let raw_instr = self
			.memory
			.read_instr(self.control.next_addr, tracker)
			.unwrap();
		let instr = Instruction::decode(byteorder::LittleEndian::read_u16(&raw_instr));
		{
			use Instruction::*;
			match instr
			{
				Call(CallVariant::Ret, offset) =>
				{
					assert_eq!(offset.value(), 0);
					self.control.ret(
						self.control.next_addr + ((offset.value() * 2) as usize),
						tracker,
					);
					// Discard everything in the ready queue
					let _ = self.operands.ready_iter(&mut self.memory, tracker);
				},
				EchoLong(offset) =>
				{
					self.operands
						.reorder_ready(offset.value() as usize, tracker);
					// Discard (now empty) ready queue
					let _ = self.operands.ready_iter(&mut self.memory, tracker);
				},
				Nop =>
				{
					// Discard ready queue
					let _ = self.operands.ready_iter(&mut self.memory, tracker);
				},
				Alu(variant, offset) =>
				{
					self.perform_alu(variant, offset, tracker);
				},
				_ => todo!(),
			}
		}
		if self.control.next_addr(&mut self.operands, tracker)
		{
			ExecResult::Ok(self)
		}
		else
		{
			ExecResult::Done(
				self.operands
					.ready_iter(&mut self.memory, tracker)
					.map(|(v, err)| {
						assert!(err.is_none());
						v
					})
					.collect::<Vec<_>>()
					.into_iter(),
			)
		}
	}

	/// Executes an Alu instruction, consuming the needed inputs and putting the
	/// result in the relevant queue.
	///
	/// The given variant is the Alu instruction to execute with the offset
	/// being which queue the result should go to (0 means the first queue after
	/// the queue of the inputs has been discarded).
	fn perform_alu(
		&mut self,
		variant: AluVariant,
		offset: Bits<5, false>,
		tracker: &mut impl MetricTracker,
	)
	{
		let (typ, mut result_scalars) = {
			let ins_count = match variant
			{
				AluVariant::Add => 2,
				AluVariant::Inc => 1,
				_ => todo!(),
			};

			// Extract operands
			let mut ins = self.operands.ready_iter(&mut self.memory, tracker);
			let mut result_scalars = Vec::new();
			let in1 = ins.next().unwrap();
			let typ = in1.0.value_type();
			let in1 = in1.0.iter();

			if ins_count == 2
			{
				let in2 = ins.next().unwrap();
				let in2 = in2.0.iter();

				for (sc1, sc2) in in1.zip(in2)
				{
					result_scalars.push(Scalar::Val(
						Self::alu_add_saturated(sc1, sc2, typ).into_boxed_slice(),
					));
				}
			}
			else if ins_count == 1
			{
				for sc1 in in1
				{
					result_scalars.push(Scalar::Val(
						Self::alu_increment_wrapping(sc1, typ).into_boxed_slice(),
					));
				}
			}
			else
			{
				todo!()
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
				// If the input both have the same sign (both positive or both
				// negative), then overflow occurs if and only if the result has
				// the opposite sign.
				// Source: https://www.doc.ic.ac.uk/~eedwards/compsys/arithmetic/index.html (2022-07-14)
				let signed_1 = in1.bytes().unwrap().last().unwrap() & 0b10000000 != 0;
				let signed_2 = in2.bytes().unwrap().last().unwrap() & 0b10000000 != 0;
				let signed_result = result_bytes.last().unwrap() & 0b10000000 != 0;

				if signed_1 && signed_2 && !signed_result
				{
					// underflow, set to lowest negative
					result_bytes.iter_mut().for_each(|b| *b = 0);
					*result_bytes.last_mut().unwrap() = 0b10000000u8;
				}
				else if !signed_1 && !signed_2 && signed_result
				{
					// overflow, set to highest value
					result_bytes.iter_mut().for_each(|b| *b = u8::MAX);
					*result_bytes.last_mut().unwrap() = 0b01111111u8;
				}
			},
			_ => (),
		}
		result_bytes
	}

	/// Performs a wrapping addition of the given scalar and 1.
	///
	/// The bytes of the input are assumed to have the given type.
	///
	/// Returns the resulting bytes of the same length as the input.
	fn alu_increment_wrapping(in1: &Scalar, typ: ValueType) -> Vec<u8>
	{
		// Create a scalar with value 1 to add to the given scalar
		let mut bytes = Vec::with_capacity(typ.scale());
		bytes.push(1);
		bytes.resize(typ.scale(), 0);
		let one_scalar = Scalar::Val(bytes.into_boxed_slice());

		Self::alu_add_carry(in1, &one_scalar).0
	}
}
