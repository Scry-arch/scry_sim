use crate::{
	control_flow::ControlFlow,
	data::{Operand, OperandStack, ProgramStack},
	memory::Memory,
	value::Value,
	ExecState, Metric, MetricTracker, Scalar, ValueType,
};
use byteorder::{ByteOrder, LittleEndian};
use duplicate::substitute;
use scry_isa::{
	Alu2OutputVariant, Alu2Variant, AluVariant, BitValue, Bits, CallVariant, Instruction, Type,
};
use std::{borrow::BorrowMut, fmt::Debug, marker::PhantomData, mem::size_of};

/// The result of performing one execution step
#[derive(Debug, Eq, PartialEq)]
pub enum ExecError
{
	/// The simulation triggered an exception
	Exception(String),

	/// The execution caused a simulation error
	Err(String),
}

/// Used to execute instructions.
#[derive(Debug)]
pub struct Executor<M: Memory, B: BorrowMut<M>>
{
	addr_space: u8, // power of 2
	control: ControlFlow,
	operands: OperandStack,
	stack: ProgramStack,
	stack_buffer: usize,
	memory: B,
	phantom: PhantomData<M>,
}
impl<M: Memory, B: BorrowMut<M>> Executor<M, B>
{
	/// Constructs an executor that starts executing from the given address from
	/// the given memory. The given values are the inputs to the instruction at
	/// start_addr
	pub fn new(
		addr_space: u8,
		start_addr: usize,
		memory: B,
		ready_ops: impl Iterator<Item = Value>,
	) -> Self
	{
		Self {
			addr_space,
			operands: OperandStack::new(ready_ops),
			control: ControlFlow::new(start_addr),
			stack: ProgramStack::new(Default::default()),
			stack_buffer: 0,
			memory,
			phantom: PhantomData,
		}
	}

	/// Constructions a new executor that is equivalent to the given execution
	/// state and uses the given memory.
	pub fn from_state(state: &ExecState, memory: B) -> Self
	{
		Self {
			addr_space: state.addr_space,
			operands: state.into(),
			control: state.into(),
			stack: state.into(),
			stack_buffer: state.stack_buffer.clone(),
			memory,
			phantom: PhantomData,
		}
	}

	/// Returns the execution state equivalent to this executors state.
	pub fn state(&self) -> ExecState
	{
		let mut frames = Vec::new();

		self.control.set_frame_state(&mut frames);
		assert!(frames.len() > 0);

		self.operands.set_all_frame_states(frames.iter_mut());
		self.stack.set_all_frame_states(frames.iter_mut());

		ExecState {
			addr_space: self.addr_space,
			address: self.control.next_addr,
			frame: frames.remove(0),
			frame_stack: frames,
			stack_buffer: self.stack_buffer.clone(),
		}
	}

	/// Discard the current ready list
	fn discard_ready_list(&mut self, tracker: &mut impl MetricTracker)
	{
		let _ = self.get_ready_iter(tracker);
	}

	/// Get the address of the current stack frames base.
	fn get_stack_base(&self) -> usize
	{
		self.stack.top().block.address
	}

	/// Get the current ready list
	fn get_ready_iter<'a>(
		&'a mut self,
		tracker: &'a mut impl MetricTracker,
	) -> impl 'a + Iterator<Item = Value>
	{
		self.operands.ready_iter(tracker)
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
			.borrow_mut()
			.read_instr(self.control.next_addr, tracker)
			.unwrap();
		let instr = Instruction::decode(byteorder::LittleEndian::read_u16(&raw_instr));
		{
			use Instruction::*;
			match instr
			{
				Call(CallVariant::Call, offset) =>
				{
					let addr = self.get_ready_iter(tracker).next().unwrap();

					let target_addr = Self::get_absolute_address(
						self.control.next_addr,
						(addr.value_type(), addr.get_first().bytes().unwrap()),
						None,
					)
					.unwrap();

					self.control.call(
						self.control.next_addr + ((offset.value() * 2) as usize),
						target_addr,
						tracker,
					);
				},
				Call(CallVariant::Ret, offset) =>
				{
					self.control.ret(
						self.control.next_addr + ((offset.value() * 2) as usize),
						tracker,
					);
					self.discard_ready_list(tracker);
				},
				EchoLong(offset) =>
				{
					self.operands
						.reorder_ready(offset.value() as usize + 1, tracker);
					// Discard (now empty) ready list
					self.discard_ready_list(tracker);
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
					self.discard_ready_list(tracker);
				},
				Echo(to_next, tar1, tar2) =>
				{
					self.operands
						.reorder(0, 1, tar2.value as usize + 1, tracker);
					self.operands
						.reorder(0, 0, tar1.value as usize + 1, tracker);
					if to_next
					{
						self.operands.reorder_ready(1, tracker);
					}
					// Discard (maybe empty) ready list
					self.discard_ready_list(tracker);
				},
				Alu(variant, offset) =>
				{
					self.perform_alu(variant, offset, tracker)?;
				},
				Alu2(variant, out, offset) =>
				{
					self.perform_alu2(variant, out, offset, tracker)?;
				},
				Constant(typ_bits, bits) =>
				{
					let typ: Type = typ_bits.try_into().unwrap();
					// Create operand from immediate and add to next list
					let mut bytes = vec![bits.value as u8];
					// The remaining bytes are sign extended if needed.
					bytes.resize(
						typ.size(),
						if typ.is_signed_int() && bits.value >= 128
						{
							u8::MAX
						}
						else
						{
							0
						},
					);

					let new_val =
						Value::singleton_typed(typ.into(), Scalar::Val(bytes.into_boxed_slice()));

					self.operands.push_operand(1, new_val.into(), tracker);

					// Forward any ready operands to the next list
					self.operands.reorder_ready(1, tracker);

					// Discard (now empty) ready list
					self.discard_ready_list(tracker);
				},
				Jump(target, location) =>
				{
					self.handle_jump(target, location, tracker);
				},
				Store =>
				{
					let next_addr = self.control.next_addr;
					let (to_store, ready_iter) = self.get_store_val(tracker)?;
					let effective_address = Self::get_mem_instr_effective_addr(
						ready_iter,
						next_addr,
						to_store.value_type().scale(),
					);

					self.store_val(tracker, &to_store, effective_address, false)?;
				},
				StoreStack(index) =>
				{
					let to_store = self.get_store_val(tracker)?.0;
					let effective_address = OperandStack::operand_stack_address(
						self.get_stack_base(),
						to_store.value_type().scale(),
						index.value as usize,
					);

					self.store_val(tracker, &to_store, Ok(effective_address), true)?;
				},
				Load(typ, offset) =>
				{
					let typ: ValueType = typ.try_into().unwrap();
					let next_addr = self.control.next_addr;
					let effective_address = {
						Self::get_mem_instr_effective_addr(
							self.get_ready_iter(tracker),
							next_addr,
							typ.scale(),
						)
						.unwrap()
					};

					self.perform_load(typ, effective_address, offset.value as usize, tracker);
				},
				LoadStack(typ, index) =>
				{
					self.discard_ready_list(tracker);
					let read_typ: ValueType = typ.try_into().unwrap();
					let effective_address = OperandStack::operand_stack_address(
						self.get_stack_base(),
						read_typ.scale(),
						index.value as usize,
					);

					self.perform_load(read_typ, effective_address, 0, tracker);
					tracker.add_stat(Metric::StackReads, 1);
					tracker.add_stat(Metric::StackReadBytes, read_typ.scale());
				},
				Pick(target) =>
				{
					let choose_first = {
						let mut ready_peek = self.operands.ready_peek();
						let condition = ready_peek.next().unwrap();
						assert!(ready_peek.next().is_some());
						assert!(ready_peek.next().is_some());
						condition
							.get_value()
							.get_first()
							.bytes()
							.unwrap()
							.iter()
							.all(|b| *b == 0)
					};

					self.operands.reorder(
						0,
						if choose_first { 1 } else { 2 },
						(target.value + 1) as usize,
						tracker,
					);

					// Consume the condition, discard the remaining
					self.get_ready_iter(tracker).next();
				},
				NoOp =>
				{
					self.discard_ready_list(tracker);
				},
				StackRes(reserving, amount, base) =>
				{
					assert_eq!(
						self.operands.ready_peek().count(),
						0,
						"TODO: reserve/free with operand"
					);

					let reserve_bytes = 2usize.pow(amount.value as u32);

					if reserving
					{
						let (base_increase, total_increase) = if base
						{
							let total_free =
								self.stack.top().block.size - self.stack.top().base_size;
							let total_missing = reserve_bytes.saturating_sub(total_free);
							(reserve_bytes, total_missing)
						}
						else
						{
							(0, reserve_bytes)
						};

						assert!(
							self.stack_buffer >= total_increase,
							"TODO: Reserve inadequate buffer"
						);

						self.stack_buffer -= total_increase;
						self.stack.top_mut().block.size += total_increase;
						self.stack.top_mut().base_size += base_increase;

						tracker.add_stat(Metric::StackReserveTotal, (!base) as usize);
						tracker.add_stat(Metric::StackReserveBase, base as usize);
						tracker.add_stat(Metric::StackReserveTotalBytes, total_increase);
						tracker.add_stat(Metric::StackReserveBaseBytes, base_increase);
					}
					else
					{
						let (base_decrease, total_decrease) = if base
						{
							(reserve_bytes, 0)
						}
						else
						{
							let total_free =
								self.stack.top().block.size - self.stack.top().base_size;
							let total_missing = reserve_bytes.saturating_sub(total_free);
							(total_missing, reserve_bytes)
						};

						assert!(
							self.stack.top().block.size >= total_decrease,
							"TODO: support stack free exception"
						);

						self.stack_buffer += total_decrease;
						self.stack.top_mut().block.size -= total_decrease;
						self.stack.top_mut().base_size -= base_decrease;

						tracker.add_stat(Metric::StackFreeTotal, (!base) as usize);
						tracker.add_stat(Metric::StackFreeBase, base as usize);
						tracker.add_stat(Metric::StackFreeTotalBytes, total_decrease);
						tracker.add_stat(Metric::StackFreeBaseBytes, base_decrease);
					}

					self.discard_ready_list(tracker);
				},
				StackAddr(size, index) =>
				{
					self.discard_ready_list(tracker);
					let effective_address = OperandStack::operand_stack_address(
						self.get_stack_base(),
						2usize.pow(size.value as u32),
						index.value as usize,
					);
					let result_type = ValueType::Uint(self.addr_space);

					self.operands.push_operand(
						0,
						Value::singleton_typed(
							result_type,
							Scalar::from_sized(effective_address, result_type.scale()),
						)
						.into(),
						tracker,
					);
				},
				instr =>
				{
					dbg!(instr);
					todo!()
				},
			}
		}
		if self
			.control
			.next_addr(&mut self.operands, &mut self.stack, tracker)?
		{
			Ok(self)
		}
		else
		{
			Err(ExecError::Err("Empty call stack".into()))
		}
	}

	/// Loads a value of the given type from memory at the given address.
	///
	/// Outputs the loaded value to the operand queue at the given offset.
	fn perform_load(
		&mut self,
		typ: ValueType,
		effective_address: usize,
		offset: usize,
		tracker: &mut impl MetricTracker,
	)
	{
		let mut loaded = Value::new_nar_typed(typ, 0);
		if let Err((_err, _addr)) =
			self.memory
				.borrow_mut()
				.read_data(effective_address, &mut loaded, 1, tracker)
		{
			todo!()
		}
		else
		{
			self.operands
				.push_operand(offset, Operand::from(loaded), tracker);
		}
	}

	/// Stores the given value at the given address (if available)
	/// If the address is a None error, will store nothing, if its an error with
	/// a messge, will return that as an error
	fn store_val(
		&mut self,
		tracker: &mut impl MetricTracker,
		to_store: &Value,
		effective_address: Result<usize, Option<ExecError>>,
		is_stack: bool,
	) -> Result<(), ExecError>
	{
		match (to_store.get_first(), effective_address)
		{
			(Scalar::Nan, _) | (_, Err(None)) => Ok(()), // Do nothing
			(Scalar::Val(_), Ok(address)) =>
			{
				self.memory
					.borrow_mut()
					.write(address, &to_store, tracker)
					.map_err(|err| ExecError::Exception(format!("Memory write error: {:?}", err)))
					.map(|result| {
						if is_stack
						{
							tracker.add_stat(Metric::StackWriteBytes, to_store.size());
							tracker.add_stat(Metric::StackWrites, 1);
						}
						result
					})
			},
			_ => Err(ExecError::Exception("Cannot store".into())),
		}
	}

	/// Gets the operand to store from the ready queue, returning the operand
	/// and the rest of the ready queue.
	fn get_store_val<'a>(
		&'a mut self,
		tracker: &'a mut impl MetricTracker,
	) -> Result<(Value, impl 'a + Iterator<Item = Value>), ExecError>
	{
		let mut ready_iter = self.get_ready_iter(tracker);

		let to_store = ready_iter.next().ok_or(ExecError::Exception(
			"Store instruction got no operands".into(),
		))?;

		Ok((to_store, ready_iter))
	}

	/// Returns the effective address of a memory instruction (load, store).
	///
	/// Extracts address resolution operands from the given ready queue
	/// iterator. If the base address is relative, uses the given current
	/// address as base. If indexed address, uses the given scale for the
	/// indices during resolution.
	///
	/// If any operand is Nar, or the base address operand is missing, returns
	/// an error with exception. If any operand is Nan, returns an error with
	/// None. Otherwise, returns the resolved address.
	fn get_mem_instr_effective_addr(
		mut ready_iter: impl Iterator<Item = Value>,
		current_addr: usize,
		scale: usize,
	) -> Result<usize, Option<ExecError>>
	{
		ready_iter
			.next()
			.ok_or(Some(ExecError::Exception(
				"Memory instruction missing first operand".into(),
			)))
			.and_then(|in1| {
				let in2 = ready_iter.next();
				let in2_extracted = in2
					.as_ref()
					.map(|in2| (in2.value_type(), in2.get_first().bytes().unwrap(), scale));

				match (in1.get_first(), in2.as_ref().map(|v| v.get_first()))
				{
					(Scalar::Nan, _) | (_, Some(Scalar::Nan)) =>
					{
						// Nothing should be done
						Err(None)
					},
					(Scalar::Val(_), None) | (Scalar::Val(_), Some(Scalar::Val(_))) =>
					{
						Self::get_absolute_address(
							current_addr,
							(in1.value_type(), in1.get_first().bytes().unwrap()),
							in2_extracted,
						)
						.ok_or(Some(ExecError::Exception(
							"Failed to calculate absolute address".into(),
						)))
					},
					_ =>
					{
						// If any is NaR, exception
						Err(Some(ExecError::Exception(
							"Memory instruction got NaR operand".into(),
						)))
					},
				}
			})
	}

	/// Converts the given bytes to the equivalent 128-bit bytes.
	/// If the given typ is a signed integer, will sign-extend the bytes
	fn extend_bytes_to_128(bytes: &[u8], typ: ValueType) -> [u8; size_of::<u128>()]
	{
		let mut new_bytes = [0u8; size_of::<u128>()];
		for (idx, byte) in bytes.iter().enumerate()
		{
			new_bytes[idx] = *byte;
			// If signed and negative, must sign extend
			if let ValueType::Int(_) = typ
			{
				if *bytes.last().unwrap() > (i8::MAX as u8)
				{
					new_bytes
						.iter_mut()
						.skip(bytes.len())
						.for_each(|b| *b = u8::MAX);
				}
			}
		}
		new_bytes
	}

	/// Returns the absolute address that the given address and its type
	/// represent.
	///
	/// If the address is unsigned, it is returned as is.
	/// If it is signed, it is treated as an offset from the current address
	/// being executed.
	/// `with_offset`, if given is the index offset to add to the resolved base
	/// address. The type and bytes are the index value to use and the last
	/// usize is the size of each element the index defines
	fn get_absolute_address(
		current_addr: usize,
		base_address: (ValueType, &[u8]),
		add_offset: Option<(ValueType, &[u8], usize)>,
	) -> Option<usize>
	{
		let address_bytes = Self::extend_bytes_to_128(base_address.1, base_address.0);
		let address = if let ValueType::Uint(_) = base_address.0
		{
			// LittleEndian lacks a read_usize, so improvise
			LittleEndian::read_u128(&address_bytes) as usize
		}
		else
		{
			// LittleEndian lacks a read_isize, so improvise
			((current_addr as i128) + LittleEndian::read_i128(&address_bytes)) as usize
		};

		let to_add = add_offset.map_or(0, |(typ, bytes, scale)| {
			if let ValueType::Int(_) = typ
			{
				unimplemented!()
			}
			LittleEndian::read_u128(&Self::extend_bytes_to_128(bytes, typ)) as usize * scale
		});

		address.checked_add(to_add)
	}

	/// Executes the jump instruction.
	fn handle_jump(
		&mut self,
		target: Bits<7, true>,
		location: Bits<6, false>,
		tracker: &mut impl MetricTracker,
	)
	{
		let (op1, op2) = {
			let mut ready_iter = self.get_ready_iter(tracker);
			(ready_iter.next(), ready_iter.next())
		};
		assert!(op2.is_none()); // TODO: implement 2-operand jumps

		// If no operands are given, it's an unconditional jump
		let unconditional = op1.is_none();
		let is_zero = op1.map_or(false, |val1| {
			val1.get_first().bytes().unwrap().iter().all(|b| *b == 0)
		});

		if target.value <= 0 && (!is_zero || unconditional)
		{
			self.control.branch(
				self.control.next_addr + (location.value * 2) as usize,
				self.control.next_addr - (target.value * -1 * 2) as usize,
				tracker,
			);
		}
		else if target.value > 0 && (is_zero || unconditional)
		{
			self.control.branch(
				self.control.next_addr + (location.value * 2) as usize,
				self.control.next_addr + ((target.value + location.value + 1) * 2) as usize,
				tracker,
			);
		}
		else
		{
			// Branch not taken
		};
	}

	fn fail_if_nan_nar(v: &Value) -> Result<(), ExecError>
	{
		match v.get_first()
		{
			Scalar::Nan | Scalar::Nar(_) => Err(ExecError::Exception("found Nan or Nar".into())),
			_ => Ok(()),
		}
	}

	fn fail_if_diff_types(v1: ValueType, v2: ValueType) -> Result<(), ExecError>
	{
		if v1 != v2
		{
			Err(ExecError::Exception("Mismatched types".into()))
		}
		else
		{
			Ok(())
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
	) -> Result<(), ExecError>
	{
		let (typ, mut result_scalars) = {
			use AluVariant::*;

			// Extract operands
			let mut ins = self.get_ready_iter(tracker);
			let mut result_scalars = Vec::new();
			let in1 = ins.next();

			let typ = match variant
			{
				Add | Sub | BitAnd | BitOr | Equal | LessThan | GreaterThan | BitXor | IsNar | NarTo =>
				{
					// Variants with 2 inputs
					let in2 = ins.next();
					let in1 = in1.ok_or(ExecError::Exception(
						"Alu instruction missing first operand".into(),
					))?;
					Self::fail_if_nan_nar(&in1)?;
					let typ = in1.value_type();

					let in2 = in2.unwrap_or_else(|| {
						let mut scalars: Vec<_> = in1
							.iter()
							.map(|_| {
								let mut bytes = Vec::new();
								bytes.resize(typ.scale(), 0);
								match variant
								{
									Add | Sub => bytes[0] = 1,                      // implicit 1
									Equal | LessThan | GreaterThan => bytes[0] = 0, // implicit 0
									x =>
									{
										dbg!(x);
										todo!()
									},
								}
								Scalar::Val(bytes.into_boxed_slice())
							})
							.collect();
						Value::new_typed(typ, scalars.remove(0), scalars).unwrap()
					});

					Self::fail_if_nan_nar(&in2)?;
					Self::fail_if_diff_types(typ, in2.value_type())?;

					let in1 = in1.iter();
					let in2 = in2.iter();

					let func = match variant
					{
						Add => Self::alu_add_saturated,
						Sub => Self::alu_sub_saturated,
						BitAnd => Self::alu_bitwise_and,
						BitOr => Self::alu_bitwise_or,
						Equal => Self::alu_equal,
						LessThan => Self::alu_less_than,
						GreaterThan => Self::alu_greater_than,
						_ =>
						{
							return Err(ExecError::Err(
								"Unsupported ALU (double input) instruction".into(),
							))
						},
					};

					let typ_out = match variant
					{
						Add | Sub | BitAnd | BitOr => typ,
						Equal | LessThan | GreaterThan => ValueType::Uint(0),
						_ =>
						{
							return Err(ExecError::Err(
								"Unsupported ALU (double input) instruction".into(),
							))
						},
					};

					for (sc1, sc2) in in1.zip(in2)
					{
						result_scalars.push(Scalar::Val(func(sc1, sc2, typ).into_boxed_slice()));
					}
					typ_out
				},
				RotateLeft | RotateRight =>
				{
					// Variants with 1 input
					let in1 = in1.ok_or(ExecError::Exception(
						"Alu instruction missing operand".into(),
					))?;
					Self::fail_if_nan_nar(&in1)?;
					let typ = in1.value_type();
					let in1 = in1.iter();
					let func = match variant
					{
						RotateLeft => Self::alu_rotate_left_once,
						RotateRight => Self::alu_rotate_right_once,
						_ =>
						{
							return Err(ExecError::Err(
								"Unsupported ALU (single input) instruction".into(),
							))
						},
					};

					for sc1 in in1
					{
						result_scalars.push(Scalar::Val(func(sc1, typ).into_boxed_slice()));
					}
					typ
				},
			};

			(typ, result_scalars)
		};
		self.operands.push_operand(
			offset.value as usize,
			Value::new_typed(typ, result_scalars.remove(0), result_scalars)
				.unwrap()
				.into(),
			tracker,
		);
		Ok(())
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
	) -> Result<(), ExecError>
	{
		let (out_typ_1, out_typ_2, mut result_scalars_low, mut result_scalars_high) = {
			// Extract operands
			let mut ins = self.get_ready_iter(tracker);
			let in1 = ins.next();
			let in1 = in1.ok_or(ExecError::Exception(
				"Alu2 instruction missing first operand".into(),
			))?;
			Self::fail_if_nan_nar(&in1)?;
			let typ = in1.value_type();

			let in2 = ins.next();
			let in2 = in2.unwrap_or({
				let mut scalars: Vec<_> = in1
					.iter()
					.map(|scal| {
						let mut bytes = Vec::new();
						bytes.resize(typ.scale(), 0);
						match variant
						{
							Alu2Variant::Add => bytes[0] = 1, // implicit 1
							Alu2Variant::Sub => bytes[0] = 1, // implicit 1
							Alu2Variant::ShiftLeft => bytes[0] = 1, // implicit 1
							Alu2Variant::ShiftRight => bytes[0] = 1, // implicit 1
							Alu2Variant::Multiply => bytes.clone_from_slice(scal.bytes().unwrap()), // implicit in1
							Alu2Variant::Division => unimplemented!(),
						}
						Scalar::Val(bytes.into_boxed_slice())
					})
					.collect();
				Value::new_typed(typ, scalars.remove(0), scalars).unwrap()
			});
			Self::fail_if_nan_nar(&in2)?;
			Self::fail_if_diff_types(typ, in2.value_type())?;

			let in1 = in1.iter();
			let in2 = in2.iter();

			let func = match variant
			{
				Alu2Variant::Add => Self::alu_add_overflowing,
				Alu2Variant::Sub => Self::alu_sub_overflowing,
				Alu2Variant::Multiply => Self::alu_multiply,
				Alu2Variant::ShiftLeft => unimplemented!(),
				Alu2Variant::ShiftRight => unimplemented!(),
				Alu2Variant::Division => unimplemented!(),
			};
			let (out_typ_1, out_typ_2) = match variant
			{
				Alu2Variant::Add => (typ, ValueType::new::<u8>()),
				Alu2Variant::Sub => (typ, ValueType::new::<u8>()),
				Alu2Variant::Multiply => (typ, typ),
				Alu2Variant::ShiftLeft => unimplemented!(),
				Alu2Variant::ShiftRight => unimplemented!(),
				Alu2Variant::Division => unimplemented!(),
			};

			let mut result_scalars_low = Vec::new();
			let mut result_scalars_high = Vec::new();
			for (sc1, sc2) in in1.zip(in2)
			{
				let (low, high) = func(sc1, sc2, typ);
				result_scalars_low.push(low);
				result_scalars_high.push(high);
			}
			(
				out_typ_1,
				out_typ_2,
				result_scalars_low,
				result_scalars_high,
			)
		};

		let low = Value::new_typed(out_typ_1, result_scalars_low.remove(0), result_scalars_low)
			.unwrap()
			.into();
		let high = Value::new_typed(
			out_typ_2,
			result_scalars_high.remove(0),
			result_scalars_high,
		)
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
		Ok(())
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

	/// Performs equality comparison.
	///
	/// The bytes of the inputs are assumed to have the given type.
	///
	/// Returns the resulting bytes. Assumes the two given scalars are valid
	/// values (not Nan or Nar) and have the same length. The result is a single
	/// byte.
	fn alu_equal(in1: &Scalar, in2: &Scalar, _: ValueType) -> Vec<u8>
	{
		vec![in1
			.bytes()
			.unwrap()
			.iter()
			.zip(in2.bytes().unwrap().iter())
			.all(|(b1, b2)| *b1 == *b2) as u8]
	}

	/// Performs less than comparison.
	///
	/// The bytes of the inputs are assumed to have the given type.
	///
	/// Returns the resulting bytes. Assumes the two given scalars are valid
	/// values (not Nan or Nar) and have the same length. The result is a single
	/// byte.
	fn alu_less_than(in1: &Scalar, in2: &Scalar, typ: ValueType) -> Vec<u8>
	{
		if let ValueType::Int(_) = typ
		{
			let neg1 = in1.bytes().unwrap().last().unwrap() >= &0b10000000u8;
			let neg2 = in2.bytes().unwrap().last().unwrap() >= &0b10000000u8;
			if neg1 && !neg2
			{
				return vec![true as u8];
			}
			else if !neg1 && neg2
			{
				return vec![false as u8];
			}
		}

		for (b1, b2) in in1
			.bytes()
			.unwrap()
			.iter()
			.rev()
			.zip(in2.bytes().unwrap().iter().rev())
		{
			if *b1 < *b2
			{
				return vec![true as u8];
			}
			else if *b1 > *b2
			{
				return vec![false as u8];
			}
			// if equal, check next byte
		}
		// all equal
		vec![false as u8]
	}

	/// Performs greater than comparison.
	///
	/// The bytes of the inputs are assumed to have the given type.
	///
	/// Returns the resulting bytes. Assumes the two given scalars are valid
	/// values (not Nan or Nar) and have the same length. The result is a single
	/// byte.
	fn alu_greater_than(in1: &Scalar, in2: &Scalar, typ: ValueType) -> Vec<u8>
	{
		if let ValueType::Int(_) = typ
		{
			let neg1 = in1.bytes().unwrap().last().unwrap() >= &0b10000000u8;
			let neg2 = in2.bytes().unwrap().last().unwrap() >= &0b10000000u8;
			if neg1 && !neg2
			{
				return vec![false as u8];
			}
			else if !neg1 && neg2
			{
				return vec![true as u8];
			}
		}

		for (b1, b2) in in1
			.bytes()
			.unwrap()
			.iter()
			.rev()
			.zip(in2.bytes().unwrap().iter().rev())
		{
			if *b1 < *b2
			{
				return vec![false as u8];
			}
			else if *b1 > *b2
			{
				return vec![true as u8];
			}
			// if equal, check next byte
		}
		// all equal
		vec![false as u8]
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

	/// Performs a shift left by 1 of the given scalar.
	///
	/// The bytes of the input are assumed to have the given type.
	///
	/// Returns the resulting bytes of the same length as the input.
	fn alu_shift_left_once(in1: &Scalar, _: ValueType) -> Vec<u8>
	{
		let mut result_bytes = Vec::new();
		let bytes = in1.bytes().unwrap();
		let mut carry = false;
		for i in 0..bytes.len()
		{
			let byte = bytes[i];
			// capture the highest order bit
			let new_carry = byte >= 0b10000000u8;
			// shift
			let mut shifted = byte << 1;
			if carry
			{
				shifted += 1;
			}
			carry = new_carry;
			result_bytes.push(shifted);
		}

		result_bytes
	}

	/// Performs a shift right by 1 of the given scalar.
	///
	/// The bytes of the input are assumed to have the given type.
	///
	/// Returns the resulting bytes of the same length as the input.
	fn alu_shift_right_once(in1: &Scalar, typ: ValueType) -> Vec<u8>
	{
		let mut result_bytes = Vec::new();
		let bytes = in1.bytes().unwrap();

		for i in 0..bytes.len()
		{
			let byte = bytes[i];
			// This is a logical shift right
			let shifted = byte >> 1;

			let high_bit = if let Some(next_byte) = bytes.get(i + 1)
			{
				// We have to ensure that the highest bit is the same as the lowest in the next
				// byte
				(*next_byte & 0b1) << 7
			}
			else
			{
				// Last byte, must ensure using arithmetic shift if signed
				if let ValueType::Int(_) = typ
				{
					byte & 0b10000000
				}
				else
				{
					0
				}
			};
			// shifted is guaranteed to have 0 in the highest bit
			result_bytes.push(high_bit + shifted);
		}

		result_bytes
	}

	/// Performs a bitwise operation on the 2 given scalars.
	///
	/// The given closure should perform the needed operand on a pair of bytes.
	///
	/// Returns the resulting bytes. Assumes the two given scalars are valid
	/// values (not Nan or Nar) and have the same length. The result then has
	/// the same length.
	fn alu_bitwise_op(in1: &Scalar, in2: &Scalar, op: impl Fn(&u8, &u8) -> u8) -> Vec<u8>
	{
		let mut result_bytes = Vec::new();

		for (b1, b2) in in1.bytes().unwrap().iter().zip(in2.bytes().unwrap().iter())
		{
			result_bytes.push(op(b1, b2));
		}
		result_bytes
	}

	/// Performs a bitwise 'and' on the given scalars.
	///
	/// Ignores the given type.
	///
	/// Returns the resulting bytes. Assumes the two given scalars are valid
	/// values (not Nan or Nar) and have the same length. The result then has
	/// the same length.
	fn alu_bitwise_and(in1: &Scalar, in2: &Scalar, _: ValueType) -> Vec<u8>
	{
		Self::alu_bitwise_op(in1, in2, |b1, b2| b1 & b2)
	}

	/// Performs a bitwise 'or' on the given scalars.
	///
	/// Ignores the given type.
	///
	/// Returns the resulting bytes. Assumes the two given scalars are valid
	/// values (not Nan or Nar) and have the same length. The result then has
	/// the same length.
	fn alu_bitwise_or(in1: &Scalar, in2: &Scalar, _: ValueType) -> Vec<u8>
	{
		Self::alu_bitwise_op(in1, in2, |b1, b2| b1 | b2)
	}

	/// Performs a rotate-left by 1 of the given scalar.
	///
	/// Ignores the given type.
	///
	/// Returns the resulting bytes of the same length as the input.
	fn alu_rotate_left_once(in1: &Scalar, _: ValueType) -> Vec<u8>
	{
		let bytes = in1.bytes().unwrap();

		bytes
			.iter()
			.fold(
				(bytes.last().unwrap() >> 7, Vec::new()),
				|(carry, mut result), b| {
					result.push((b << 1) + carry);
					(b >> 7, result)
				},
			)
			.1
	}

	/// Performs a rotate-right by 1 of the given scalar.
	///
	/// Ignores the given type.
	///
	/// Returns the resulting bytes of the same length as the input.
	fn alu_rotate_right_once(in1: &Scalar, _: ValueType) -> Vec<u8>
	{
		let bytes = in1.bytes().unwrap();

		let mut result = bytes
			.iter()
			.rev()
			.fold((bytes[0] << 7, Vec::new()), |(carry, mut result), b| {
				result.push((b >> 1) + carry);
				(b << 7, result)
			})
			.1;
		result.reverse();
		result
	}

	/// Performs multiplication on the given scalars, returning the lower-order
	/// result
	fn alu_multiply(sc1: &Scalar, sc2: &Scalar, typ: ValueType) -> (Scalar, Scalar)
	{
		use ValueType::*;
		let result: Vec<u8> = substitute! {
			[throw_away [];] // used just to allow duplicate! in match case position
			match typ {
				duplicate!{
					[
						Sign	letter;
						[Uint]	[u];
						[Int]	[i];
					]
					Sign(0) => {
						paste::paste!{
							let v1 = (sc1.bytes().unwrap()[0] as [<letter 8>]) as [<letter 16>];
							let v2 = (sc2.bytes().unwrap()[0] as [<letter 8>]) as [<letter 16>];
							let mut result = [0u8;2];
							LittleEndian::[< write_ letter 16 >]
										(&mut result, v1 * v2);
							result.into()
						}
					},
					duplicate!{
						[
							power size bits bits2;
							[1] [2] [16] [32];
							[2] [4] [32] [64];
							[3] [8] [64] [128];
						]
						Sign(power) => {
							paste::paste!{
								let v1 = LittleEndian::
									[< read_ letter bits >](sc1.bytes().unwrap()) as [< letter bits2 >];
								let v2 = LittleEndian::
									[< read_ letter bits >](sc2.bytes().unwrap()) as [< letter bits2 >];
								let mut result = [0u8;size*2];
								LittleEndian::[< write_ letter bits2 >]
									(&mut result, v1 * v2);
								result.into()
							}
						},
					}
				}
				_ => todo!()
			}
		};

		let (low, high) = result.split_at(result.len() / 2);
		(
			Scalar::Val(low.iter().cloned().collect::<Vec<_>>().into_boxed_slice()),
			Scalar::Val(high.iter().cloned().collect::<Vec<_>>().into_boxed_slice()),
		)
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
