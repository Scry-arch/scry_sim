use byteorder::{ByteOrder, LittleEndian};
use duplicate::duplicate_item;
use quickcheck_macros::quickcheck;
use scry_isa::Instruction;
use scry_sim::{
	Event, EventTracker, ExecState, IdEvent, MemError, Memory, OperandEvent, OperandList,
	OperandQueue, Scalar, Value, ValueType,
};
use std::{
	cmp::max,
	collections::{HashMap, HashSet},
	mem::size_of,
};

#[derive(Debug)]
pub struct OneShotTracker
{
	pub events: Vec<Event>,
	pub next_id: usize,
}

fn vec_eq_unordered<T: PartialEq + Clone>(v1: &Vec<T>, v2: &Vec<T>) -> bool
{
	let mut v2_clone = v2.clone();
	for e in v1.clone()
	{
		if let Some((idx, _)) = v2_clone.iter().enumerate().find(|(_, e2)| **e2 == e)
		{
			v2_clone.remove(idx);
		}
		else
		{
			// Not found
			return false;
		}
	}
	v2_clone.is_empty()
}

impl OneShotTracker
{
	pub fn new() -> OneShotTracker
	{
		Self {
			events: vec![],
			next_id: 0,
		}
	}

	pub fn new_id(&mut self) -> usize
	{
		let id = self.next_id;
		self.next_id += 1;
		id
	}

	/// Returns whether the given list of events is equivalent to the current
	/// list of events.
	///
	/// Even IDs are allowed to be different as long as an equivalent mapping
	/// can be found between self's and other's IDs
	pub fn equivalent(&self, other: &Self) -> bool
	{
		if self.events.len() != other.events.len()
		{
			return false;
		}

		// First check that all non-id events are equivalent
		let self_no_id = self
			.events
			.iter()
			.filter(|e| !matches!(e, Event::Id(_, _)))
			.cloned()
			.collect::<Vec<_>>();
		let other_no_id = other
			.events
			.iter()
			.filter(|e| !matches!(e, Event::Id(_, _)))
			.cloned()
			.collect::<Vec<_>>();

		if !vec_eq_unordered(&self_no_id, &other_no_id)
		{
			return false;
		}

		let group_by_id = |mut acc: HashMap<usize, HashSet<IdEvent>>, ev: &Event| {
			if let Event::Id(id, ty) = ev
			{
				let was_new = acc.entry(*id).or_insert(HashSet::new()).insert(ty.clone());
				assert!(was_new);
			}
			acc
		};
		let self_id_groups = self.events.iter().fold(HashMap::new(), group_by_id);
		let mut other_id_groups = other.events.iter().fold(HashMap::new(), group_by_id);

		for (_, group) in self_id_groups
		{
			if let Some(k) = other_id_groups
				.iter()
				.find(|(_, other_group)| **other_group == group)
				.map(|(k, _)| *k)
			{
				other_id_groups.remove(&k);
			}
			else
			{
				// Could not find equivalent id group
				return false;
			}
		}
		if other_id_groups.len() > 0
		{
			// Has group without equivalent in self
			return false;
		}

		true
	}
}

impl<F: IntoIterator<Item = Event>> From<F> for OneShotTracker
{
	fn from(events: F) -> Self
	{
		let events: Vec<Event> = events.into_iter().collect();
		// Find the highest id
		let next_id = events.iter().fold(0, |acc, ev| {
			match ev
			{
				Event::Id(id, _) => max(*id + 1, acc),
				_ => acc,
			}
		});

		Self { events, next_id }
	}
}

#[test]
fn one_shot_tracker_self_equivalence()
{
	let x = OneShotTracker {
		events: vec![
			Event::Instruction(0, Instruction::NoOp),
			Event::Id(0, IdEvent::Operand(OperandEvent::Consume)),
			Event::Id(1, IdEvent::Operand(OperandEvent::Consume)),
			Event::Id(2, IdEvent::Operand(OperandEvent::Produce)),
		],
		next_id: 3,
	};

	assert_eq!(x, x);
}

impl EventTracker for OneShotTracker
{
	fn new_operand_id(&mut self) -> usize
	{
		self.new_id()
	}

	fn new_control_id(&mut self) -> usize
	{
		self.new_id()
	}

	fn event(&mut self, event: Event)
	{
		self.events.push(event);
	}
}

impl PartialEq for OneShotTracker
{
	fn eq(&self, other: &Self) -> bool
	{
		self.equivalent(other)
	}
}

/// A memory that always produces the same instruction and data.
///
/// All read operations will always succeed regardless of address or alignment.
///
/// The first member is the encoding of the instruction that the memory
/// should return from every `read_instr` while the second is what
/// `read_data` will populate all read bytes with.
///
/// If `MAY_STORE` is true, writes will succeed. If not, writes will fail.
///
/// `write_raw` will do nothing.
///
/// Meant for testing.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct RepeatingMem<const MAY_STORE: bool>(pub u16, pub u8);
impl<const MAY_STORE: bool> Memory for RepeatingMem<MAY_STORE>
{
	fn read_raw(&mut self, _: usize) -> Result<u8, MemError>
	{
		Ok(self.1)
	}

	fn read_data(&mut self, _: usize, into: &mut Value, len: usize)
		-> Result<(), (MemError, usize)>
	{
		let mut scalar_bytes = Vec::with_capacity(into.scale());
		scalar_bytes.resize(into.scale(), self.1);
		let scalar = Scalar::Val(scalar_bytes.into_boxed_slice());

		let mut scalars = Vec::with_capacity(len);
		scalars.resize_with(len, || scalar.clone());
		let result = Value::new_typed(into.value_type(), scalars.remove(0), scalars).unwrap();

		*into = result;
		Ok(())
	}

	fn read_instr(&mut self, _: usize) -> Result<[u8; 2], MemError>
	{
		let mut result = [0; 2];
		LittleEndian::write_u16(&mut result, self.0);
		Ok(result)
	}

	fn write_raw(&mut self, _: usize, _: u8) -> Result<(), MemError>
	{
		Ok(())
	}

	fn write(&mut self, addr: usize, from: &Value) -> Result<(), (MemError, usize)>
	{
		if MAY_STORE
		{
			AllowWrite(self).write(addr, from)
		}
		else
		{
			Err((MemError::InvalidAddr, addr))
		}
	}
}

/// Wrapper around a memory that accept all writes and allows them to succeed
/// regardless of address.
#[derive(Debug)]
pub struct AllowWrite<'a, M: Memory>(pub &'a mut M);
impl<'a, M: Memory> Memory for AllowWrite<'a, M>
{
	delegate::delegate! {
		to self.0{
			fn read_raw(&mut self, addr: usize) -> Result<u8, MemError>;
			fn read_data(
				&mut self,
				addr: usize,
				into: &mut Value,
				len: usize,
			) -> Result<(), (MemError, usize)>;
			fn read_instr(
				&mut self,
				addr: usize,
			) -> Result<[u8; 2], MemError>;
			fn write_raw(&mut self, addr: usize, size: u8) -> Result<(), MemError>;
		}
	}

	fn write(&mut self, _: usize, _: &Value) -> Result<(), (MemError, usize)>
	{
		Ok(())
	}
}

/// Advances the given operand queue by 1, i.e. all indices reduced by 1
pub fn advance_queue(from: OperandQueue) -> OperandQueue
{
	from.into_iter().map(|(idx, ops)| (idx - 1, ops)).collect()
}

/// Adds the value to the queue at the given index
///
/// If the queue is already at the given depth, does not add the value and
/// return false. Otherwise the value is added and returns true
pub fn add_value_to_queue_idx(
	queue: &mut OperandQueue,
	op_list_depth: usize,
	idx: usize,
	v: Value,
) -> bool
{
	if let Some(list) = queue.get_mut(&idx)
	{
		if list.len() < op_list_depth
		{
			list.push(v);
			true
		}
		else
		{
			false
		}
	}
	else
	{
		queue.insert(idx, OperandList::new(v, vec![]));
		true
	}
}

/// Adds the given list of values to the queue
///
/// Stops adding once the queue reaches the given depths
/// Returns how many elements were added to the queue
pub fn add_list_to_queue_idx(
	queue: &mut OperandQueue,
	op_list_depth: usize,
	idx: usize,
	mut iter: impl Iterator<Item = Value>,
) -> usize
{
	let mut added = 0;
	while let Some(v) = iter.next()
	{
		if !add_value_to_queue_idx(queue, op_list_depth, idx, v)
		{
			break;
		}
		added += 1;
	}
	added
}

/// Adds the given list of values to the queue
///
/// Stops adding once the queue reaches the allowed depth
/// Returns how many elements were added to the queue
pub fn add_to_queue_idx(
	state: &mut ExecState,
	idx: usize,
	iter: impl Iterator<Item = Value>,
) -> usize
{
	let depth = state.op_list_depth();
	add_list_to_queue_idx(&mut state.frame.op_queue, depth, idx, iter)
}

/// Regress the given operand queue by 1, i.e. all indices increased by 1
pub fn regress_queue(from: OperandQueue) -> OperandQueue
{
	from.into_iter().map(|(idx, ops)| (idx + 1, ops)).collect()
}

/// Interprets the given scalar as a usize.
pub fn as_usize(operand: &Scalar) -> usize
{
	let mut result = 0usize;

	assert!(size_of::<usize>() >= operand.bytes().unwrap().len());

	for (idx, byte) in operand.bytes().unwrap().iter().enumerate()
	{
		let mut byte_usize = *byte as usize;
		let shift_count = idx * 8;
		byte_usize <<= shift_count;
		result += byte_usize;
	}

	result
}

/// Interprets the given scalar as a isize.
pub fn as_isize(operand: &Scalar) -> isize
{
	let mut result = 0usize;

	assert!(size_of::<usize>() >= operand.bytes().unwrap().len());

	for (idx, byte) in operand.bytes().unwrap().iter().enumerate()
	{
		let mut byte_usize = *byte as usize;
		let shift_count = idx * 8;
		byte_usize <<= shift_count;
		result += byte_usize;
	}

	// If negative, must sign extend
	if *operand.bytes().unwrap().last().unwrap() > (i8::MAX as u8)
		&& size_of::<usize>() > operand.bytes().unwrap().len()
	{
		let mut extending_bits = usize::MAX;
		extending_bits = extending_bits
			.overflowing_shl(8 * operand.bytes().unwrap().len() as u32)
			.0;
		result += extending_bits;
	}

	// Reinterpret as isize
	isize::from_le_bytes(result.to_le_bytes())
}

/// Get the address represented by the given value.
///
/// The value is assumed to be an unsigned integer.
pub fn get_absolute_address(addr: &Scalar) -> usize
{
	as_usize(addr)
}

/// Get the absolute address from the current address and the given offset
/// value.
///
/// The value is assumed to be a signed integer
pub fn get_relative_address(current_address: usize, offset: &Scalar) -> Option<usize>
{
	let offset_value = as_isize(offset);
	if offset_value < 0
	{
		current_address.checked_sub(offset_value.abs_diff(0))
	}
	else
	{
		current_address.checked_add(offset_value.abs_diff(0))
	}
}

/// Get the absolute address from the given indexed address.
///
/// The base address may me signed or unsiged while the index is assumed to be
/// unsigned. In case of relative base address, the given current address is
/// used.
pub fn get_indexed_address(
	current_address: usize,
	base_addr: &Value,
	index: &Scalar,
	index_scale: usize,
) -> Option<usize>
{
	let checked_absolute_addr = if let ValueType::Int(_) = base_addr.value_type()
	{
		get_relative_address(current_address, base_addr.get_first())
	}
	else
	{
		Some(get_absolute_address(base_addr.get_first()))
	};
	let offset = as_usize(&index).checked_mul(index_scale);

	checked_absolute_addr.and_then(|addr| offset.and_then(|offset| addr.checked_add(offset)))
}

/// Test as_usize on various type lengths
#[duplicate_item(
	name 					typ;
	[test_as_usize_u8] 		[u8];
	[test_as_usize_u16] 	[u16];
	[test_as_usize_u32]  	[u32];
	[test_as_usize_usize]	[usize];
)]
#[quickcheck]
fn name(operand: typ) -> bool
{
	let mut scal = Scalar::Nan;
	scal.set_val(&operand.to_le_bytes());
	let result = as_usize(&scal);
	result == (operand as usize)
}

/// Test as_isize on various type lengths
#[duplicate_item(
	name 					typ;
	[test_as_isize_i8] 		[i8];
	[test_as_isize_i16] 	[i16];
	[test_as_isize_i32]  	[i32];
	[test_as_isize_isize]	[isize];
)]
#[quickcheck]
fn name(operand: typ) -> bool
{
	let mut scal = Scalar::Nan;
	scal.set_val(&operand.to_le_bytes());
	let result = as_isize(&scal);
	result == (operand as isize)
}
