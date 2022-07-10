use crate::{
	control_flow::ControlFlow, data::OperandQueue, memory::Memory, value::Value, ExecState,
};
use byteorder::ByteOrder;
use scry_isa::{CallVariant, Instruction};
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
	pub fn step(mut self) -> ExecResult<impl Iterator<Item = Value> + Debug>
	{
		let raw_instr = self.memory.read_instr(self.control.next_addr).unwrap();
		let instr = Instruction::decode(byteorder::LittleEndian::read_u16(&raw_instr));
		{
			use Instruction::*;
			match instr
			{
				Call(CallVariant::Ret, offset) =>
				{
					assert_eq!(offset.value(), 0);
					self.control
						.ret(self.control.next_addr + ((offset.value() * 2) as usize));
					// Discard everything in the ready queue
					let _ = self.operands.ready_iter(&mut self.memory);
				},
				EchoLong(offset) =>
				{
					self.operands.reorder_ready(offset.value() as usize);
					// Discard (now empty) ready queue
					let _ = self.operands.ready_iter(&mut self.memory);
				},
				Nop =>
				{
					// Discard ready queue
					let _ = self.operands.ready_iter(&mut self.memory);
				},
				_ => todo!(),
			}
		}
		if self.control.next_addr(&mut self.operands)
		{
			ExecResult::Ok(self)
		}
		else
		{
			ExecResult::Done(
				self.operands
					.ready_iter(&mut self.memory)
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
