use crate::{
	control_flow::ControlFlow, data::OperandQueue, memory::Memory, value::Value, ExecState,
	MetricTracker, Scalar, ValueType,
};
use byteorder::ByteOrder;
use scry_isa::{AluVariant, CallVariant, Instruction};
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
				Alu(AluVariant::Add, offset) =>
				{
					let (typ, mut result_scalars) = {
						// Extract operands
						let mut ins = self.operands.ready_iter(&mut self.memory, tracker);
						let in1 = ins.next().unwrap();
						let typ = in1.0.value_type();
						let in1 = in1.0.iter();
						let in2 = ins.next().unwrap();
						let in2 = in2.0.iter();

						let mut result_scalars = Vec::new();
						for (sc1, sc2) in in1.zip(in2)
						{
							let mut result_bytes = Vec::new();
							let mut carry = false;

							for (b1, b2) in
								sc1.bytes().unwrap().iter().zip(sc2.bytes().unwrap().iter())
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

							// If overflow, saturate
							match (typ, carry)
							{
								(ValueType::Uint(_), true) =>
								{
									result_bytes.iter_mut().for_each(|b| *b = u8::MAX)
								},
								(ValueType::Int(_), _) =>
								{
									// If the input both have the same sign (both positive or both
									// negative), then overflow occurs if and only if the result has
									// the opposite sign.
									// Source: https://www.doc.ic.ac.uk/~eedwards/compsys/arithmetic/index.html (2022-07-14)
									let signed_1 =
										sc1.bytes().unwrap().last().unwrap() & 0b10000000 != 0;
									let signed_2 =
										sc2.bytes().unwrap().last().unwrap() & 0b10000000 != 0;
									let signed_result =
										result_bytes.last().unwrap() & 0b10000000 != 0;

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
							result_scalars.push(Scalar::Val(result_bytes.into_boxed_slice()));
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
}
