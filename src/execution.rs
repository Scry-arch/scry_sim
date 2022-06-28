use crate::{
	control_flow::ControlFlow,
	data::{OperandQueue, Value},
	memory::Memory,
};
use byteorder::ByteOrder;
use scry_isa::{CallVariant, Instruction};

pub struct Executor
{
	control: ControlFlow,
	operands: OperandQueue,
	memory: Memory,
}

/// The result of performing one execution step
pub enum ExecResult<I: Iterator<Item = Value>>
{
	/// The executor performed the step successfully
	Ok(Executor),

	/// The executor finished executing with the given result values and reports
	Done(I),

	/// The execution caused an error
	Err,
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

	/// Perform one execution step.
	pub fn execute(mut self) -> ExecResult<impl Iterator<Item = Value>>
	{
		if let Some(addr) = self.control.next_addr(&mut self.operands)
		{
			let raw_instr = self.memory.read_instr(addr).unwrap();
			let instr = Instruction::decode(byteorder::LittleEndian::read_u16(&raw_instr));
			{
				match instr
				{
					Instruction::Call(CallVariant::Ret, offset) =>
					{
						assert_eq!(offset.value(), 0);
						self.control.ret(addr + ((offset.value() * 2) as usize));
						// Discard everything in the ready queue
						let _ = self.operands.ready_iter(&mut self.memory);
					},
					Instruction::EchoLong(offset) =>
					{
						self.operands.reorder_ready(offset.value() as usize);
						// Discard (now empty) ready queue
						let _ = self.operands.ready_iter(&mut self.memory);
					},
					_ => todo!(),
				}
			}
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
