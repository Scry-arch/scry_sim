use crate::{
	Block, CallFrameState, ControlFlowType, ExecState, OperandList, Scalar, StackFrame, Value,
};
use duplicate::duplicate_item;
use num_traits::{PrimInt, Unsigned};
use quickcheck::{Arbitrary, Gen};
use std::{
	collections::{HashMap, VecDeque},
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

impl Arbitrary for InstrAddr
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		let mut addr = Arbitrary::arbitrary(g);
		// Ensure is 2-byte aligned
		if addr % 2 != 0
		{
			addr -= 1;
		}
		Self(addr)
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		Box::new(self.0.shrink().map(|mut addr| {
			// Ensure is 2-byte aligned
			if addr % 2 != 0
			{
				addr -= 1;
			}
			Self(addr)
		}))
	}
}

impl Arbitrary for ControlFlowType
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		match u8::arbitrary(g) % 100
		{
			0..=39 => Self::Branch(InstrAddr::arbitrary(g).0),
			40..=79 => Self::Call(InstrAddr::arbitrary(g).0),
			_ => Self::Return,
		}
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		match self
		{
			Self::Branch(t) => Box::new(InstrAddr(*t).shrink().map(|v| Self::Branch(v.0))),
			Self::Call(t) => Box::new(InstrAddr(*t).shrink().map(|v| Self::Call(v.0))),
			Self::Return => Box::new(std::iter::empty()),
		}
	}
}

impl Arbitrary for Block
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		loop
		{
			let address = usize::arbitrary(g);
			let size = usize::arbitrary(g) % (64 * g.size());

			let result = Self { address, size };

			if result.validate::<false>().is_ok()
			{
				return result;
			}
		}
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
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
				),
		)
	}
}

impl Arbitrary for StackFrame
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		let mut frame = StackFrame {
			block: Arbitrary::arbitrary(g),
			base_size: 0,
		};

		// decide on primary size
		if frame.block.size > 0
		{
			frame.base_size = usize::arbitrary(g) % frame.block.size;
		}

		// make sure is valid frame
		frame.validate().unwrap();

		frame
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		let clone = self.clone();
		Box::new(
			// Shrink by reducing block size
			self.block
			.shrink()
			.map(move|new_block| {
				let mut clone = clone.clone();
				clone.block = new_block;
				clone
			})
			.filter(|shrunk| shrunk.validate().is_ok())

			// Shrink by reducing primary size
			.chain(
				{
					let clone = self.clone();
					self.base_size.shrink().map(move |new_size| {
						let mut clone = clone.clone();
						clone.base_size = new_size;
						clone.validate().unwrap();
						clone
					})
				}
			),
		)
	}
}

impl Arbitrary for CallFrameState
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		let ret_addr: usize = InstrAddr::arbitrary(g).0;
		let branches: Vec<(InstrAddr, ControlFlowType)> = Arbitrary::arbitrary(g);

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

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		enum ShrinkState
		{
			Start,
			BranchTrig(usize),
			BranchTyp(usize),
			BranchRem(usize, VecDeque<usize>),
			Operand(VecDeque<usize>, usize),
			Stack,
			Done,
		}
		struct Shrinker
		{
			original: CallFrameState,
			state: ShrinkState,
			other: Box<dyn Iterator<Item = CallFrameState>>,
		}
		impl Iterator for Shrinker
		{
			type Item = CallFrameState;

			fn next(&mut self) -> Option<Self::Item>
			{
				// First empty any remaining shrinks
				if let Some(result) = self.other.next()
				{
					return Some(result);
				}

				use ShrinkState::*;
				match &mut self.state
				{
					Start =>
					{
						// Shrink return address
						let clone = self.original.clone();
						self.other =
							Box::new(InstrAddr(self.original.ret_addr).shrink().map(move |ret| {
								let mut clone = clone.clone();
								clone.ret_addr = ret.0;
								clone
							}));
						self.state = BranchTrig(0);
						self.next()
					},
					BranchTrig(idx) =>
					{
						// Shrink branch trigger address
						if let Some((addr, _)) = self.original.branches.iter().nth(*idx)
						{
							let clone = self.original.clone();
							let addr = *addr;
							self.other = Box::new(InstrAddr(addr).shrink().map(move |new_addr| {
								let mut clone = clone.clone();
								let typ = clone.branches.remove(&addr).unwrap();
								clone.branches.insert(new_addr.0, typ);
								clone
							}));
							self.state = BranchTyp(*idx);
						}
						else
						{
							self.state =
								Operand(self.original.op_queue.keys().cloned().collect(), 0);
						}
						self.next()
					},
					BranchTyp(idx) =>
					{
						// Shrink branch type
						let (addr, typ) = self.original.branches.iter().nth(*idx).unwrap();

						let clone = self.original.clone();
						let addr = *addr;
						let typ = *typ;

						self.other = Box::new(typ.shrink().map(move |new_typ| {
							let mut clone = clone.clone();
							*clone.branches.get_mut(&addr).unwrap() = new_typ;
							clone
						}));
						self.state =
							BranchRem(*idx, self.original.branches.keys().cloned().collect());
						self.next()
					},
					BranchRem(idx, addrs) =>
					{
						// Remove a branch
						if let Some(addr) = addrs.pop_front()
						{
							let mut clone = self.original.clone();
							clone.branches.remove(&addr).unwrap();
							Some(clone)
						}
						else
						{
							self.state = BranchTrig(*idx + 1);
							self.next()
						}
					},
					Operand(list_idxs, op_idx) =>
					{
						if let Some(list_idx) = list_idxs.front()
						{
							// Reduce the index of operand lists where possible
							let q_clone = if *list_idx > 0
								&& !self.original.op_queue.contains_key(&(list_idx - 1))
							{
								let mut q_clone = self.original.clone();
								let q = q_clone.op_queue.remove(list_idx).unwrap();
								q_clone.op_queue.insert(list_idx - 1, q);
								Some(q_clone)
							}
							else
							{
								None
							};

							let op_list = self.original.op_queue.get(&list_idx).unwrap();
							if let Some(v) = op_list.iter().nth(*op_idx)
							{
								// Remove operand
								let mut rem_clone = self.original.clone();
								let rem_from = rem_clone.op_queue.get_mut(list_idx).unwrap();
								if *op_idx == 0
								{
									if rem_from.rest.len() > 0
									{
										rem_from.first = rem_from.rest.remove(0);
									}
									else
									{
										// Only one operand in list, remove whole list
										rem_clone.op_queue.remove(list_idx).unwrap();
									}
								}
								else
								{
									rem_from.rest.remove(*op_idx - 1);
								}
								// Shrink value operand
								let clone = self.original.clone();
								let list_idx = *list_idx;
								let op_idx2 = *op_idx;
								self.other = Box::new(
									once(rem_clone)
										.chain(v.shrink().map(move |v| {
											let mut clone = clone.clone();
											*clone
												.op_queue
												.get_mut(&list_idx)
												.unwrap()
												.iter_mut()
												.nth(op_idx2)
												.unwrap() = v;
											clone
										}))
										.chain(q_clone.into_iter()),
								);

								*op_idx += 1;
							}
							else
							{
								// No more operands, next list
								list_idxs.pop_front().unwrap();
								*op_idx = 0;
							}
							self.next()
						}
						else
						{
							self.state = Stack;
							self.next()
						}
					},
					Stack =>
					{
						// Shrink stack frame
						let clone = self.original.clone();
						self.other = Box::new(self.original.stack.shrink().map(move |shrunk| {
							let mut clone = clone.clone();
							clone.stack = shrunk;
							clone
						}));
						self.state = Done;
						self.next()
					},
					_ => None,
				}
			}
		}

		Box::new(Shrinker {
			original: self.clone(),
			state: ShrinkState::Start,
			other: Box::new(empty()),
		})
	}
}

impl Arbitrary for ExecState
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		let frame = CallFrameState::arbitrary(g);
		let earliest_trigger = frame.branches.iter().fold(usize::MAX, |earliest, (k, _)| {
			if earliest > *k
			{
				*k
			}
			else
			{
				earliest
			}
		});

		let mut result = Self {
			// Ensure that the address may be increased by 2 (after executing an instruction)
			// without causing overflow.
			// Also ensure address precedes all branch trigger locations
			address: InstrAddr::arbitrary(g).0 % (std::cmp::min(earliest_trigger, usize::MAX - 3)),
			frame,
			frame_stack: vec![],
			stack_buffer: usize::arbitrary(g),
		};
		assert!(result.validate().is_ok());

		// We limit the call depth for performance reasons
		let stack_len = usize::arbitrary(g) % 2;
		let mut next_addr = result.frame.stack.block.address + result.frame.stack.block.size;

		// Create call frames
		for _ in 0..stack_len
		{
			loop
			{
				result.frame_stack.push(Arbitrary::arbitrary(g));
				result.frame_stack.last_mut().unwrap().stack.block.address = next_addr;

				// Check new frame does not clash with others
				if result.validate().is_ok()
				{
					let block = &result.frame_stack.last().unwrap().stack.block;
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
		let mut result = Vec::new();

		// Shrink address
		result.extend(InstrAddr(self.address).shrink().map(|new_addr| {
			let mut clone = self.clone();
			clone.address = new_addr.0;
			clone
		}));

		// Shrink first frame
		result.extend(
			self.frame
				.shrink()
				.map(|new_frame| {
					let mut clone = self.clone();
					clone.frame = new_frame;
					clone
				})
				.filter(|new_frame| new_frame.validate().is_ok()),
		);

		// Remove first frame
		if self.frame_stack.len() > 0
		{
			let mut clone = self.clone();
			clone.frame = clone.frame_stack.remove(0);
			result.push(clone);
		}

		Box::new(result.into_iter())
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
