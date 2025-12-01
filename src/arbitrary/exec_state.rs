use crate::{
	arbitrary::WeightedArbitrary, Block, CallFrameState, CallType, ControlFlowType, ExecState,
	OperandList, Scalar, StackFrame, Value,
};
use duplicate::duplicate_item;
use num_traits::{PrimInt, Unsigned};
use quickcheck::{Arbitrary, Gen};
use std::{
	collections::HashMap,
	fmt::Debug,
	iter::{empty, once},
	ops::{Add, AddAssign, Div, Sub},
};

/// Integers that are much more likely to be close to zero than far from it
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct SmallInt<N: PrimInt + Sub + Add + Div + Arbitrary>(pub N);
impl<N: PrimInt + AddAssign + Debug + Unsigned + Arbitrary> Arbitrary for SmallInt<N>
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		let mut x = N::zero();
		let mut limit = 0;
		loop
		{
			match u8::arbitrary(g) % 100
			{
				0..=1 => break,
				_ =>
				{
					if x.to_usize().unwrap() == g.size()
					{
						x = N::zero();
						limit += 1;
					}
					else
					{
						x += N::one();
					}
				},
			}
			// Ensure we don't loop forever
			if limit == 100
			{
				break;
			}
		}
		Self(x)
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		Box::new(self.0.shrink().map(|x| Self(x)))
	}
}

/// An address that is guaranteed to be 2-byte aligned
#[derive(Debug, Clone, Copy)]
pub struct InstrAddr(pub usize);

/// The weight is the maximum pointer in the address space.
///
/// The returned instruction will be smaller than it.
impl WeightedArbitrary<usize> for InstrAddr
{
	fn arbitrary_weighted(g: &mut Gen, max_addr: usize) -> Self
	{
		let mut addr = usize::arbitrary(g) % max_addr;
		// Ensure is 2-byte aligned
		if addr % 2 != 0
		{
			addr -= 1;
		}
		assert!(addr <= max_addr);
		Self(addr)
	}

	fn shrink_weighted(&self, max_addr: usize) -> Box<dyn Iterator<Item = Self>>
	{
		Box::new(self.0.shrink().map(move |mut addr| {
			// Ensure is 2-byte aligned
			if addr % 2 != 0
			{
				addr -= 1;
			}
			assert!(addr <= max_addr);
			Self(addr)
		}))
	}
}
impl Arbitrary for InstrAddr
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		InstrAddr::arbitrary_weighted(g, usize::MAX)
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		InstrAddr::shrink_weighted(self, usize::MAX)
	}
}

/// The weight is the maximum pointer in the address space.
///
/// Any returned address with be lower than the max.
impl WeightedArbitrary<usize> for ControlFlowType
{
	fn arbitrary_weighted(g: &mut Gen, max_addr: usize) -> Self
	{
		match u8::arbitrary(g) % 100
		{
			0..=39 => Self::Branch(InstrAddr::arbitrary_weighted(g, max_addr).0),
			40..=79 => Self::Call(InstrAddr::arbitrary_weighted(g, max_addr).0),
			_ => Self::Return,
		}
	}

	fn shrink_weighted(&self, max_addr: usize) -> Box<dyn Iterator<Item = Self>>
	{
		match self
		{
			Self::Branch(t) =>
			{
				Box::new(
					InstrAddr(*t)
						.shrink_weighted(max_addr)
						.map(|v| Self::Branch(v.0)),
				)
			},
			Self::Call(t) =>
			{
				Box::new(
					InstrAddr(*t)
						.shrink_weighted(max_addr)
						.map(|v| Self::Call(v.0)),
				)
			},
			Self::Return => Box::new(empty()),
		}
	}
}
impl Arbitrary for ControlFlowType
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		ControlFlowType::arbitrary_weighted(g, usize::MAX)
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		ControlFlowType::shrink_weighted(self, usize::MAX)
	}
}

/// The weight is the maximum pointer in the address space.
///
/// Any returned address with be lower than the max.
impl WeightedArbitrary<usize> for Block
{
	fn arbitrary_weighted(g: &mut Gen, max_addr: usize) -> Self
	{
		loop
		{
			let address = usize::arbitrary(g) % max_addr;
			let available_size = max_addr - address;
			let size = (usize::arbitrary(g) % (64 * g.size())) % available_size;

			let result = Self { address, size };

			if result.validate::<false>().is_ok()
			{
				return result;
			}
		}
	}

	fn shrink_weighted(&self, _: usize) -> Box<dyn Iterator<Item = Self>>
	{
		let clone = self.clone();
		// Shrink size
		Box::new(
			self.size
				.shrink()
				.filter(|shrunk| *shrunk > 0)
				.map(move |shrunk| {
					Self {
						address: clone.address,
						size: shrunk,
					}
				})
				.chain(
					// Shrink address
					self.address.shrink().map(move |address| {
						Self {
							address,
							size: clone.size,
						}
					}),
				)
				.map(|shrunk| {
					assert!(shrunk.validate::<false>().is_ok());
					shrunk
				}),
		)
	}
}
impl Arbitrary for Block
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		Block::arbitrary_weighted(g, usize::MAX)
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		Block::shrink_weighted(self, usize::MAX)
	}
}

/// The weight is the maximum pointer in the address space.
///
/// Any returned address with be lower than the max.
impl WeightedArbitrary<usize> for StackFrame
{
	fn arbitrary_weighted(g: &mut Gen, max_addr: usize) -> Self
	{
		let mut frame = StackFrame {
			block: WeightedArbitrary::arbitrary_weighted(g, max_addr),
			base_size: 0,
		};

		if frame.block.size > 0
		{
			frame.base_size = usize::arbitrary(g) % frame.block.size;
		}

		// make sure is valid frame
		frame.validate().unwrap();

		frame
	}

	fn shrink_weighted(&self, max_addr: usize) -> Box<dyn Iterator<Item = Self>>
	{
		let clone1 = self.clone();
		let clone2 = self.clone();
		Box::new(
			// Shrink by reducing primary size
			self.base_size
				.shrink()
				.map(move |new_size| {
					let mut clone = clone1.clone();
					clone.base_size = new_size;
					clone.validate().unwrap();
					clone
				})
				.chain(
					// Shrink by reducing block size
					self.block
						.shrink_weighted(max_addr)
						.map(move |new_block| {
							let mut clone = clone2.clone();
							clone.block = new_block;
							clone
						})
						.filter(|shrunk| shrunk.validate().is_ok()),
				),
		)
	}
}
impl Arbitrary for StackFrame
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		StackFrame::arbitrary_weighted(g, usize::MAX)
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		StackFrame::shrink_weighted(self, usize::MAX)
	}
}

/// The weight is the maximum pointer in the address space.
///
/// Any returned address with be lower than the max.
impl WeightedArbitrary<usize> for CallFrameState
{
	fn arbitrary_weighted(g: &mut Gen, max_addr: usize) -> Self
	{
		let ret_addr: usize = InstrAddr::arbitrary_weighted(g, max_addr).0;
		let mut branches = Vec::new();

		for _ in 0..SmallInt::<usize>::arbitrary(g).0
		{
			branches.push((
				InstrAddr::arbitrary_weighted(g, max_addr),
				ControlFlowType::arbitrary_weighted(g, max_addr),
			));
		}

		let mut op_queue = HashMap::new();

		// How many operand lists
		for i in 0..SmallInt::<usize>::arbitrary(g).0
		{
			let op_count = SmallInt::<u8>::arbitrary(g).0 % 5;
			if op_count > 1
			{
				let mut ops = Vec::new();
				for _ in 0..op_count
				{
					ops.push(Arbitrary::arbitrary(g));
				}
				op_queue.insert(i, OperandList::new(ops.remove(0), ops));
			}
		}

		Self {
			ret_addr,
			branches: branches
				.into_iter()
				.filter(|(addr, _)| addr.0 != 0) // disallow branch triggers at 0
				.map(|(trig, ctrl)| (trig.0, ctrl))
				.collect(),
			op_queue,
			stack: Arbitrary::arbitrary(g),
		}
	}

	fn shrink_weighted(&self, max_addr: usize) -> Box<dyn Iterator<Item = Self>>
	{
		let clone1 = self.clone();
		let clone2 = self.clone();
		let clone3 = self.clone();
		let clone4 = self.clone();
		let branches = self.branches.clone();
		let op_queue = self.op_queue.clone();
		Box::new(
			// Shrink return address
			InstrAddr(self.ret_addr)
				.shrink_weighted(max_addr)
				.map(move |shrunk| {
					let mut clone = clone1.clone();
					clone.ret_addr = shrunk.0;
					clone
				})
				.chain(
					// Shrink branches
					branches.into_iter().flat_map(move |(addr, typ)| {
						let mut branch_clone1 = clone2.clone();
						branch_clone1.branches.remove_entry(&addr);
						let branch_clone2 = branch_clone1.clone();
						let branch_clone3 = branch_clone1.clone();
						// Shrink trigger address
						InstrAddr(addr).shrink_weighted(max_addr).filter_map(move|shrunk|{
						let mut clone = branch_clone1.clone();
						if clone.branches.insert(shrunk.0, typ).is_none() {
							Some(clone)
						} else {
							// Discard if branch triggers collide
							None
						}
					})
					.chain({
						// Shrink trigger type
						typ.shrink_weighted(max_addr).map(move|shrunk|{
							let mut clone = branch_clone2.clone();
							clone.branches.insert(addr, shrunk);
							clone
						})
					})
					// Shrink by removing the branch
					.chain(once(branch_clone3))
					}),
				)
				.chain(op_queue.into_iter().flat_map(move |(idx, list)| {
					let list_len = list.len();
					let list_clone = list.clone();
					let list_clone2 = list.clone();
					let mut removed_clone = clone3.clone();
					removed_clone.op_queue.remove_entry(&idx);
					let removed_clone2 = removed_clone.clone();
					let queue_clone1 = clone3.clone();
					let queue_clone2 = clone3.clone();

					// Remove whole list
					once(removed_clone.clone())
						.chain(
							// Remove operand from list
							if list_len > 1 { Some(()) } else { None }
								.into_iter()
								.flat_map(move |_| {
									let list_clone2 = list_clone2.clone();
									let queue_clone2 = removed_clone2.clone();
									(0..list_len).map(move |idx| {
										let mut list_clone3 = list_clone2.clone();
										if idx == 0
										{
											if let Some(new_first) = list_clone2.rest.first()
											{
												list_clone3.first = new_first.clone();
												list_clone3.rest.remove(0);
											}
										}
										else
										{
											list_clone3.rest.remove(idx - 1);
										}
										let mut clone = queue_clone2.clone();
										clone.op_queue.insert(idx, list_clone3);
										clone
									})
								}),
						)
						.chain(
							// Shrink operand list indices
							idx.shrink().filter_map(move |shrunk| {
								if removed_clone.op_queue.contains_key(&shrunk)
								{
									None
								}
								else
								{
									let mut clone = removed_clone.clone();
									clone.op_queue.insert(shrunk, list_clone.clone());
									Some(clone)
								}
							}),
						)
						.chain(
							// Shrink operand values

							// Shrink first value
							clone3.op_queue[&idx].first.shrink().map(move|shrunk| {
							let mut clone = queue_clone1.clone();
							clone.op_queue.get_mut(&idx).unwrap().first = shrunk;
							clone
						})
						// Shrink the rest
						.chain({
							let clone3 = clone3.clone();
							(1..list_len).flat_map(move|list_idx| {
								let queue_clone2 = queue_clone2.clone();
								clone3.op_queue[&idx].rest[list_idx-1].shrink().map(move|shrunk| {
									let mut clone = queue_clone2.clone();
									clone.op_queue.get_mut(&idx).unwrap().rest[list_idx-1] = shrunk;
									clone
								})
							})
						}
						),
						)
				}))
				.chain(
					// Shrink stack frame
					self.stack.shrink_weighted(max_addr).map(move |shrunk| {
						let mut clone = clone4.clone();
						clone.stack = shrunk;
						clone
					}),
				),
		)
	}
}
impl Arbitrary for CallFrameState
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		CallFrameState::arbitrary_weighted(g, usize::MAX)
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		CallFrameState::shrink_weighted(self, usize::MAX)
	}
}

impl<T: Arbitrary> Arbitrary for CallType<T>
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		match u8::arbitrary(g) % 3
		{
			0 => CallType::Call,
			1 => CallType::Trap,
			_ => CallType::Interrupt(Arbitrary::arbitrary(g)),
		}
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		match self
		{
			CallType::Call => Box::new(empty()),
			CallType::Trap => Box::new(once(CallType::Call)),
			CallType::Interrupt(v) =>
			{
				Box::new(
					v.shrink()
						.map(move |shrunk| CallType::Interrupt(shrunk))
						.chain(once(CallType::Call)),
				)
			},
		}
	}
}

impl Arbitrary for ExecState
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		let addr_space = u8::arbitrary(g) % 3;
		let mut result = Self {
			addr_space,
			// Ensure that the address may be increased by 2 (after executing an instruction)
			// without causing overflow.
			// Also ensure address precedes all branch trigger locations
			address: 0,
			foli: Arbitrary::arbitrary(g),
			frame: CallFrameState {
				// Dummy frame
				ret_addr: 0,
				branches: Default::default(),
				op_queue: Default::default(),
				stack: Default::default(),
			},
			frame_stack: vec![],
			stack_buffer: 0,
		};
		let max_addr = result.max_addr();

		result.frame = CallFrameState::arbitrary_weighted(g, max_addr);
		result.stack_buffer = usize::arbitrary(g) % max_addr;

		let earliest_trigger = result
			.frame
			.branches
			.iter()
			.fold(usize::MAX, |earliest, (k, _)| {
				if earliest > *k
				{
					*k
				}
				else
				{
					earliest
				}
			});
		result.address =
			InstrAddr::arbitrary_weighted(g, std::cmp::min(earliest_trigger, max_addr - 3)).0;
		assert!(result.validate().is_ok());

		// We limit the call depth for performance reasons
		let stack_len = usize::arbitrary(g) % 2;
		let mut next_addr = result.frame.stack.block.address + result.frame.stack.block.size;

		// Create call frames
		for _ in 0..stack_len
		{
			loop
			{
				result.frame_stack.push((
					CallFrameState::arbitrary_weighted(g, max_addr),
					Arbitrary::arbitrary(g),
				));
				result.frame_stack.last_mut().unwrap().0.stack.block.address = next_addr;

				// Check new frame does not clash with others
				if result.validate().is_ok()
				{
					let block = &result.frame_stack.last().unwrap().0.stack.block;
					next_addr = block.address + block.size;
					break;
				}
				else
				{
					// Try again
					result.frame_stack.pop();
				}
			}
		}

		assert!(result.validate().is_ok());
		result
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		let clone = self.clone();
		let clone2 = self.clone();
		let clone3 = self.clone();
		// Shrink by removing the last stack frame
		let without_last_frame = if self.frame_stack.len() <= 1
		{
			None
		}
		else
		{
			let mut clone = self.clone();
			clone.frame_stack.pop();
			Some(clone)
		};
		Box::new(
			without_last_frame
				.into_iter()
				.chain(
					// Shrink address
					InstrAddr(self.address).shrink().map(move |new_addr| {
						let mut clone = clone.clone();
						clone.address = new_addr.0;
						clone
					}),
				)
				.chain(
					// Shrink first frame
					self.frame
						.shrink()
						.map(move |new_frame| {
							let mut clone = clone2.clone();
							clone.frame = new_frame;
							clone
						})
						.filter(|new_frame| new_frame.validate().is_ok()),
				)
				.chain(
					// Remove first frame
					if self.frame_stack.len() > 0
					{
						let mut clone = clone3.clone();
						clone.frame = clone.frame_stack.remove(0).0;
						Some(clone)
					}
					else
					{
						None
					}
					.into_iter(),
				)
				.chain(
					// Shrink foli
					self.foli.shrink().map(move |v| {
						let mut clone = clone3.clone();
						clone.foli = v;
						clone
					}),
				),
		)
	}
}

/// Helper trait for types that can be used to generate arbitrary execution
/// states.
pub trait Restriction: Clone + Debug + AsMut<ExecState> + AsRef<ExecState> + 'static
{
	/// Each restriction takes an inner execution state that it restricts.
	/// A raw ExecState is the base restriction (with no restriction)
	type Inner: Restriction;

	/// Checks whether the inner state uphold the restriction this type defines
	fn restriction_holds(inner: &Self::Inner) -> bool;

	/// Updates the given state to uphold the restriction.
	fn conform(state: &mut Self::Inner, g: &mut Gen);

	/// When shrinking the state, this method may add custom shrinking based on
	/// this restriction.
	///
	/// By default, no custom shrinking is done.
	fn add_shrink(_state: &Self::Inner) -> Box<dyn Iterator<Item = Self>>
	{
		Box::new(empty())
	}
}
impl Restriction for ExecState
{
	type Inner = Self;

	fn restriction_holds(_: &Self::Inner) -> bool
	{
		true
	}

	fn conform(_: &mut Self::Inner, _: &mut Gen) {}
}
impl AsRef<Self> for ExecState
{
	fn as_ref(&self) -> &Self
	{
		self
	}
}
impl AsMut<Self> for ExecState
{
	fn as_mut(&mut self) -> &mut Self
	{
		self
	}
}

/// Define "restriction" types used to generate arbitrary execution state with
/// some restrictions, e.g. can't have any MustRead operands or must have some
/// number of operands.
///
/// The structs defined here automatically get implemented the arbitrary trait,
/// assuming the "Restriction" trait is implemented for them somewhere else.
#[duplicate_item(
	[
		mod_name [no_cf]
		name [NoCF]
		desc [
			"Use to generate arbitrary states that don't have a control flow trigger
			after the next instruction"
		]
		extra_generics []
		extra_generics_pass []
	]
	[
		mod_name [limited_ops]
		name [LimitedOps]
		desc [
			"Use to generate arbitrary states with upper/lower limits on operands going to \
			the next instruction. Both limits are inclusive."
		]
		extra_generics [ const NEXT_OP_MIN: usize, const NEXT_OP_MAX: usize ]
		extra_generics_pass [NEXT_OP_MIN, NEXT_OP_MAX]
	]
	[
		mod_name [simple_ops]
		name [SimpleOps]
		desc [
			"Use to generate arbitrary states whose operands going to the next instruction are simple:\n \
			 * Have the same type \n \
			 * Have the same len \n \
			 * Not have any Nar or Nan\n "
		]
		extra_generics []
		extra_generics_pass []
	]
)]
mod mod_name
{
	use super::*;

	#[doc = desc]
	#[derive(Clone, Debug)]
	pub struct name<T: Restriction, extra_generics>(pub T);
	impl<T: Restriction, extra_generics> name<T, extra_generics_pass>
	{
		pub fn restrict(inner: T) -> Option<Self>
		{
			if Self::restriction_holds(&inner)
			{
				Some(Self(inner))
			}
			else
			{
				None
			}
		}
	}
	impl<T: Restriction + Arbitrary, extra_generics> Arbitrary for name<T, extra_generics_pass>
	{
		fn arbitrary(g: &mut Gen) -> Self
		{
			let mut state: T = Arbitrary::arbitrary(g);

			Self::conform(&mut state, g);

			Self(state)
		}

		fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
		{
			let filtered = self.0.shrink().filter_map(|state| {
				// Remove any read operand for the next instruction
				Self::restrict(state)
			});
			let extra_shrinks = Self::add_shrink(&self.0);
			Box::new(filtered.chain(extra_shrinks))
		}
	}
	impl<T: Restriction, extra_generics> AsRef<ExecState> for name<T, extra_generics_pass>
	{
		fn as_ref(&self) -> &ExecState
		{
			self.0.as_ref()
		}
	}
	impl<T: Restriction, extra_generics> AsMut<ExecState> for name<T, extra_generics_pass>
	{
		fn as_mut(&mut self) -> &mut ExecState
		{
			self.0.as_mut()
		}
	}
}
pub use self::{limited_ops::*, no_cf::*, simple_ops::*};

impl<T: Restriction> Restriction for NoCF<T>
{
	type Inner = T;

	fn restriction_holds(inner: &Self::Inner) -> bool
	{
		!inner
			.as_ref()
			.frame
			.branches
			.contains_key(&inner.as_ref().address)
	}

	fn conform(state: &mut Self::Inner, _: &mut Gen)
	{
		// Remove any control flow that would trigger after next instructions
		let addr = state.as_ref().address;
		let _ = state.as_mut().frame.branches.remove(&addr);
	}
}
impl<T: Restriction, const NEXT_OP_MIN: usize, const NEXT_OP_MAX: usize> Restriction
	for LimitedOps<T, NEXT_OP_MIN, NEXT_OP_MAX>
{
	type Inner = T;

	fn restriction_holds(inner: &Self::Inner) -> bool
	{
		// Ensure number of operands is between the limits
		let next_op_count = inner
			.as_ref()
			.frame
			.op_queue
			.get(&0)
			.map_or(0, OperandList::len);
		next_op_count >= NEXT_OP_MIN && next_op_count <= NEXT_OP_MAX
	}

	fn conform(state: &mut Self::Inner, g: &mut Gen)
	{
		let next_op_count = state
			.as_ref()
			.frame
			.op_queue
			.get(&0)
			.map_or(0, OperandList::len);
		// Ensure minimum is upheld
		if next_op_count < NEXT_OP_MIN
		{
			let mut ops_to_add = NEXT_OP_MIN - next_op_count;
			if NEXT_OP_MIN < NEXT_OP_MAX
			{
				ops_to_add += usize::arbitrary(g) % (NEXT_OP_MAX - NEXT_OP_MIN);
			}
			if ops_to_add > 0
			{
				if state.as_ref().frame.op_queue.get(&0).is_none()
				{
					let op1 = Value::arbitrary(g);
					state
						.as_mut()
						.frame
						.op_queue
						.insert(0, OperandList::new(op1, Vec::new()));
					ops_to_add -= 1;
				}
				if ops_to_add > 0
				{
					for _ in 0..ops_to_add
					{
						let op = Value::arbitrary(g);
						state.as_mut().frame.op_queue.get_mut(&0).unwrap().push(op);
					}
				}
			}
		}
		else if next_op_count > NEXT_OP_MAX
		{
			if NEXT_OP_MAX == 0
			{
				state.as_mut().frame.op_queue.remove(&0).unwrap();
			}
			else
			{
				state
					.as_mut()
					.frame
					.op_queue
					.get_mut(&0)
					.unwrap()
					.rest
					.resize_with(NEXT_OP_MAX - 1, || unreachable!());
			}
		}
	}
}
impl<T: Restriction> Restriction for SimpleOps<T>
{
	type Inner = T;

	fn restriction_holds(inner: &Self::Inner) -> bool
	{
		if inner.as_ref().frame.op_queue.contains_key(&0)
		{
			let op_list = inner.as_ref().frame.op_queue.get(&0).unwrap();
			let len = op_list.first.len();
			let typ = op_list.first.value_type();
			op_list.iter().all(|v| {
				v.len() == len && v.value_type() == typ && v.iter().all(|sc| sc.bytes().is_some())
			})
		}
		else
		{
			true
		}
	}

	fn conform(state: &mut Self::Inner, g: &mut Gen)
	{
		if state.as_ref().frame.op_queue.contains_key(&0)
		{
			let v = &state.as_ref().frame.op_queue.get(&0).unwrap().first;
			let (len, typ) = (v.len(), v.value_type());
			let frame = &mut state.as_mut().frame;
			let op_list = frame.op_queue.get_mut(&0).unwrap();
			// Convert all operands to the same type/length of the first
			for v in op_list.iter_mut()
			{
				let mut scalars = Vec::new();
				for _ in 0..len
				{
					let mut bytes = Vec::new();
					for _ in 0..typ.scale()
					{
						bytes.push(Arbitrary::arbitrary(g));
					}
					scalars.push(Scalar::Val(bytes.into_boxed_slice()));
				}
				*v = Value::new_typed(typ, scalars.remove(0), scalars).unwrap();
			}
		}
	}

	fn add_shrink(_state: &Self::Inner) -> Box<dyn Iterator<Item = Self>>
	{
		let typ = _state
			.as_ref()
			.frame
			.op_queue
			.get(&0)
			.map(|op_list| op_list.first.value_type());
		let self_clone = _state.clone();
		Box::new(typ.into_iter().flat_map(move |typ| {
			let self_clone = self_clone.clone();
			typ.shrink().map(move |new_typ| {
				let mut clone = self_clone.clone();
				let clone_frame = &mut clone.as_mut().frame;
				for v in clone_frame.op_queue.get_mut(&0).unwrap().iter_mut()
				{
					let mut new_scalars = Vec::with_capacity(new_typ.scale());
					if new_typ.scale() < typ.scale()
					{
						for sc in v.iter()
						{
							let new_bytes: Vec<_> = sc.bytes().unwrap()[0..new_typ.scale()]
								.iter()
								.cloned()
								.collect();
							new_scalars.push(Scalar::Val(new_bytes.into_boxed_slice()));
						}
					}
					else
					{
						new_scalars.extend(v.iter().cloned());
					}
					*v = Value::new_typed(new_typ, new_scalars.remove(0), new_scalars).unwrap();
				}
				Self(clone)
			})
		}))
	}
}
