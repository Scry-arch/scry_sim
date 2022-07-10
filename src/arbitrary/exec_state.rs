use crate::{CallFrameState, ControlFlowType, ExecState, OperandState, Value};
use duplicate::*;
use num_traits::{PrimInt, Unsigned};
use quickcheck::{Arbitrary, Gen};
use std::{
	collections::{HashMap, VecDeque},
	fmt::Debug,
	iter::once,
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

impl Arbitrary for CallFrameState
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		let ret_addr: usize = InstrAddr::arbitrary(g).0;
		let branches: Vec<(InstrAddr, ControlFlowType)> = Arbitrary::arbitrary(g);

		let mut op_queues = HashMap::new();
		let mut reads = Vec::new();

		// How many operand queues
		for i in 0..SmallInt::<usize>::arbitrary(g).0
		{
			let op_count = SmallInt::<u8>::arbitrary(g).0 % 5;
			if op_count > 1
			{
				let mut ops = Vec::new();
				for _ in 0..op_count
				{
					match u8::arbitrary(g) % 100
					{
						0..=79 => ops.push(OperandState::Ready(Arbitrary::arbitrary(g))),
						_ =>
						{
							match u8::arbitrary(g) % 100
							{
								0..=79 | _ if reads.len() == 0 =>
								{
									// New int not dependent on others
									reads.push((
										Arbitrary::arbitrary(g),
										SmallInt::arbitrary(g).0,
										Arbitrary::arbitrary(g),
									));
									ops.push(OperandState::MustRead(reads.len() - 1));
								},
								_ =>
								{
									ops.push(OperandState::MustRead(
										usize::arbitrary(g) % reads.len(),
									));
								},
							}
						},
					}
				}
				op_queues.insert(i, (ops.remove(0), ops));
			}
		}

		Self {
			ret_addr,
			branches: branches
				.into_iter()
				.map(|(trig, ctrl)| (trig.0, ctrl))
				.collect(),
			op_queues,
			reads,
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
			ReadsAddr(usize),
			ReadsLen(usize),
			ReadsTyp(usize),
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
				// We wrap the match in duplicate to allow use to duplicate enum variant matches
				duplicate! {[throwaway [];]
					match &mut self.state {
						Start => {
							// Shrink return address
							let clone = self.original.clone();
							self.other = Box::new(InstrAddr(self.original.ret_addr).shrink().map(move|ret| {
								let mut clone = clone.clone();
								clone.ret_addr = ret.0;
								clone
							}));
							self.state = BranchTrig(0);
							self.next()
						}
						BranchTrig(idx) => {
							// Shrink branch trigger address
							if let Some((addr, _)) = self.original.branches.iter().nth(*idx) {

								let clone = self.original.clone();
								let addr = *addr;
								self.other = Box::new(
									InstrAddr(addr).shrink().map(move|new_addr| {
										let mut clone = clone.clone();
										let typ = clone.branches.remove(&addr).unwrap();
										clone.branches.insert(new_addr.0, typ);
										clone
									})
								);
								self.state = BranchTyp(*idx);
							} else {
								self.state = Operand(self.original.op_queues.keys().cloned().collect(),0);
							}
							self.next()
						}
						BranchTyp(idx) => {
							// Shrink branch type
							let (addr, typ) = self.original.branches.iter().nth(*idx).unwrap();

							let clone = self.original.clone();
							let addr = *addr;
							let typ = *typ;

							self.other = Box::new(
								typ.shrink().map(move|new_typ| {
									let mut clone = clone.clone();
									*clone.branches.get_mut(&addr).unwrap() = new_typ;
									clone
								}));
							self.state = BranchRem(*idx, self.original.branches.keys().cloned().collect());
							self.next()
						}
						BranchRem(idx, addrs) => {
							// Remove a branch
							if let Some(addr) = addrs.pop_front()
							{
								let mut clone = self.original.clone();
								clone.branches.remove(&addr).unwrap();
								Some(clone)
							} else {
								self.state = BranchTrig(*idx +1);
								self.next()
							}
						}
						Operand(q_idxs,op_idx) => {
							if let Some(q_idx) = q_idxs.front() {
								// Reduce the index of queues where possible
								let q_clone = if *q_idx>0 && !self.original.op_queues.contains_key(&(q_idx-1)) {
									let mut q_clone = self.original.clone();
									let q = q_clone.op_queues.remove(q_idx).unwrap();
									q_clone.op_queues.insert(q_idx-1, q);
									Some(q_clone)
								} else {
									None
								};

								let (q_first, q_rest) = self.original.op_queues.get(&q_idx).unwrap();
								if let Some(op) = once(q_first).chain(q_rest.iter()).nth(*op_idx) {
									// Remove operand
									let mut rem_clone = self.original.clone();
									let (op_first, op_rest) = rem_clone.op_queues.get_mut(q_idx).unwrap();
									if *op_idx == 0 {
										if op_rest.len() > 0{
											*op_first = op_rest.remove(0);
										} else {
											// Only one operand in queue, remove whole queue
											rem_clone.op_queues.remove(q_idx).unwrap().0;
										}
									} else {
										op_rest.remove(*op_idx-1);
									}
									rem_clone.clean_reads();

									match op {
										OperandState::Ready(v) => {
											// Shrink value operand
											let clone = self.original.clone();
											let q_idx = *q_idx;
											let op_idx2 = *op_idx;
											self.other = Box::new(once(rem_clone).chain(v.shrink().map(move|v| {
												let mut clone = clone.clone();
												let (first, rest) = clone
													.op_queues
													.get_mut(&q_idx)
													.unwrap();
												*once(first).chain(rest.iter_mut())
													.nth(op_idx2)
													.unwrap() = OperandState::Ready(v);
												clone
											})).chain(q_clone.into_iter()));
										}
										OperandState::MustRead(read_idx) => {
											// Disconnect from other MustReads by cloning the read and referencing it instead
											let clone1 = if self.original.count_read_refs(*read_idx) > 1 {
												let mut clone = self.original.clone();
												let (first, rest) = clone
													.op_queues
													.get_mut(q_idx)
													.unwrap();
												*once(first).chain(rest.iter_mut())
													.nth(*op_idx)
													.unwrap() = OperandState::MustRead(read_idx+1);
												clone.reads.push(self.original.reads.get(*read_idx).unwrap().clone());
												Some(clone)
											} else {
												None
											};

											// Convert to simple ready value
											let mut clone2 = self.original.clone();
											let (first, rest) = clone2
												.op_queues
												.get_mut(q_idx)
												.unwrap();
											*once(first).chain(rest.iter_mut())
												.nth(*op_idx)
												.unwrap() = OperandState::Ready(Value::new_nan::<u8>());
											clone2.clean_reads();

											self.other = Box::new([rem_clone, clone2].into_iter().chain(clone1.into_iter()).chain(q_clone.into_iter()));
										}
									}

									*op_idx += 1;
									self.next()
								} else {
									// No more operands, next queue
									q_idxs.pop_front().unwrap();
									*op_idx = 0;
									self.next()
								}
							} else {
								self.state = ReadsAddr(0);
								self.next()
							}
						}
						duplicate!{[
								variant		tuple_idx	update_to;
								[ReadsAddr]	[0]			[ReadsLen(*read_idx)];
								[ReadsLen]	[1]			[ReadsTyp(*read_idx)];
								[ReadsTyp]	[2]			[ReadsAddr(*read_idx+1)];
							]
							variant(read_idx) => {
								if let Some(read_tup) = self.original.reads.get(*read_idx) {
									let clone = self.original.clone();
									let read_idx2 = *read_idx;
									let shrunks = read_tup.tuple_idx.shrink().map(move |shrunk| {
										let mut clone = clone.clone();
										clone.reads.get_mut(read_idx2).unwrap().tuple_idx = shrunk;
										clone
									});
									self.other = Box::new(shrunks);
									self.state = update_to;
									self.next()
								} else {
									// No more reads
									self.state = Done;
									self.next()
								}
							}
						}

						_ => None
					}
				}
			}
		}

		Box::new(Shrinker {
			original: self.clone(),
			state: ShrinkState::Start,
			other: Box::new(std::iter::empty()),
		})
	}
}

impl Arbitrary for ExecState
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		// We limit the call depth for performance reasons
		let stack_len = usize::arbitrary(g) % 5;
		let mut stack = Vec::with_capacity(stack_len);
		for _ in 0..stack_len
		{
			stack.push(Arbitrary::arbitrary(g));
		}

		Self {
			// Ensure that the address may be increased by 2 (after executing an instructino)
			// without causing overflow
			address: InstrAddr::arbitrary(g).0 % (usize::MAX - 3),
			frame: Arbitrary::arbitrary(g),
			frame_stack: stack,
		}
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		let mut result = Vec::new();

		// Shrink first frame
		result.extend(self.frame.shrink().map(|new_frame| {
			let mut clone = self.clone();
			clone.frame = new_frame;
			clone
		}));

		// Remove first frame
		if self.frame_stack.len() > 0
		{
			let mut clone = self.clone();
			clone.frame = clone.frame_stack.remove(0);
			result.push(clone);
		}

		// Shrink/remove other frames
		for (idx, frame) in self.frame_stack.iter().enumerate()
		{
			// Shrink frames
			result.extend(frame.shrink().map(|f| {
				let mut clone = self.clone();
				*clone.frame_stack.get_mut(idx).unwrap() = f;
				clone
			}));
			// Remove frames
			let mut clone = self.clone();
			clone.frame_stack.remove(idx);
			result.push(clone)
		}

		Box::new(result.into_iter())
	}
}

/// Use to generate an ExecState that is guaranteed to not have control flow
/// that will trigger right after the next instruction.
#[derive(Clone, Debug)]
pub struct NoCFExecState(pub ExecState);

impl Arbitrary for NoCFExecState
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		let mut state = ExecState::arbitrary(g);

		// Remove any control flow that would trigger after next instructions
		let _ = state.frame.branches.remove(&state.address);

		Self(state)
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		Box::new(self.0.shrink().filter_map(|state| {
			if state.frame.branches.contains_key(&state.address)
			{
				None
			}
			else
			{
				Some(Self(state))
			}
		}))
	}
}
