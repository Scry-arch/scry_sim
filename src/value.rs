use byteorder::{ByteOrder, LittleEndian};
use duplicate::duplicate_item;
use num_traits::PrimInt;
use scry_isa::{Bits, Type};
use std::{cmp::max, iter::once};

/// The type of a value
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ValueType
{
	/// Unsigned integer with power of 2 bytes.
	///
	/// E.g. `Uint(0)` is an unsigned 1 byte integer, `Uint(1)` is an unsigned 2
	/// byte integer, and so on.
	Uint(u8),

	/// Signed integer with power of 2 bytes.
	///
	/// E.g. `Int(0)` is an signed 1 byte integer, `Int(1)` is an unsigned 2
	/// byte integer, and so on.
	Int(u8),
}
impl ValueType
{
	/// Constructs a value type equivalent to the given type parameter.
	///
	/// E.g. `new::<u8>()` constructs `Uint(0)`, `new::<i16>()` constructs
	/// `Int(1)`.
	pub fn new<N: PrimInt>() -> Self
	{
		let size = std::mem::size_of::<N>();
		assert!(size <= u8::MAX as usize);
		assert_eq!(size.count_ones(), 1);
		let pow2_size = size.trailing_zeros();
		if N::min_value() < N::zero()
		{
			ValueType::Int(pow2_size as u8)
		}
		else
		{
			ValueType::Uint(pow2_size as u8)
		}
	}

	/// Returns the power of 2 scale of the type.
	///
	/// I.e., u8 has a power of 0, u16 is 1, i16 is also 1, i32 is 2, etc.
	pub fn power(&self) -> u8
	{
		match self
		{
			ValueType::Uint(x) | ValueType::Int(x) => *x,
		}
	}

	/// Returns the number of bytes one scalar element if this type takes up.
	pub fn scale(&self) -> usize
	{
		match self
		{
			ValueType::Uint(x) | ValueType::Int(x) => 2usize.pow(*x as u32),
		}
	}

	/// Returns whether this type is a signed integer type
	pub fn is_signed_integer(&self) -> bool
	{
		match self
		{
			ValueType::Uint(_) => false,
			ValueType::Int(_) => true,
		}
	}

	#[duplicate_item(
		typ;
		[u8];
		[u16];
		[u32];
		[u64];
		[i8];
		[i16];
		[i32];
		[i64];
	)]
	/// Returns the ValueType representation of the given type
	pub fn typ() -> ValueType
	{
		let log2 = size_of::<typ>().ilog2() as u8;
		if stringify!(typ).starts_with("u")
		{
			ValueType::Uint(log2)
		}
		else
		{
			ValueType::Int(log2)
		}
	}

	/// Returns the effective type between this and the other.
	pub fn get_effective_type(&self, other: &Self) -> Self
	{
		let unsigned = !self.is_signed_integer() || !other.is_signed_integer();
		let pow = max(self.power(), other.power());
		if unsigned
		{
			ValueType::Uint(pow)
		}
		else
		{
			ValueType::Int(pow)
		}
	}
}
impl From<Type> for ValueType
{
	fn from(t: Type) -> Self
	{
		if t.is_signed_int()
		{
			Self::Int(t.size_pow2())
		}
		else
		{
			Self::Uint(t.size_pow2())
		}
	}
}
impl From<ValueType> for Type
{
	fn from(t: ValueType) -> Self
	{
		match t
		{
			ValueType::Uint(size) => Type::Uint(size),
			ValueType::Int(size) => Type::Int(size),
		}
	}
}
impl TryFrom<Bits<4, false>> for ValueType
{
	type Error = ();

	fn try_from(value: Bits<4, false>) -> Result<Self, Self::Error>
	{
		Ok(TryInto::<Type>::try_into(value)?.into())
	}
}

/// A scalar piece of data.
///
/// Each scalar is either a Nar, a Nan, or has a value which matches its
/// implicit type.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Scalar
{
	/// Proper value with given data.
	Val(Box<[u8]>),

	/// Not-A-Result: Signifies something went wrong when this value was to be
	/// produced.
	///
	/// Comes with a payload, which is currently undefined
	Nar(usize),

	/// Not-A-Number: Signifies the intentional lack of a value.
	Nan,
}
impl Scalar
{
	/// Sets the value's state.
	///
	/// If not already a proper `Val` value, becomes one.
	/// The given bytes are the new value's state.
	/// Care should be taken to ensure the length of the slice matches the
	/// value's type.
	pub fn set_val(&mut self, bytes: &[u8])
	{
		if let Scalar::Val(val) = self
		{
			val.iter_mut().zip(bytes).for_each(|(v, b)| *v = *b);
		}
		else
		{
			*self = Scalar::Val(Vec::from(bytes).into_boxed_slice());
		}
	}

	pub fn bytes(&self) -> Option<&[u8]>
	{
		match self
		{
			Scalar::Val(bytes) => Some(bytes),
			_ => None,
		}
	}

	/// Returns a scalar containing the given value and of the given size.
	pub fn from_sized(val: usize, size: usize) -> Self
	{
		assert!(size.is_power_of_two());
		let mut bytes = val.to_le_bytes().into_iter().take(size).collect::<Vec<_>>();
		bytes.resize(size, 0);
		assert_eq!(bytes.len(), size);

		Self::Val(bytes.into_boxed_slice())
	}

	/// Extends the given scalar to the given scale and returns the result.
	///
	/// Will sign- or zero-extend based on the given type
	///
	/// If the scalar is not a value, returns a copy of the scalar
	pub fn extend(&self, scale: usize, typ: &ValueType) -> Self
	{
		match self
		{
			Scalar::Val(bytes) =>
			{
				let negative = bytes.iter().last().map_or(false, |b| b >= &0b1000_0000);
				let extend_with = if negative && typ.is_signed_integer()
				{
					u8::MAX
				}
				else
				{
					0
				};
				let mut new_bytes: Vec<_> = bytes.iter().cloned().collect();
				new_bytes.resize(scale, extend_with);
				Scalar::Val(new_bytes.into_boxed_slice())
			},
			_ => self.clone(),
		}
	}

	#[duplicate_item(
		name			typ		value_typ(scale);
		[u128_value]	[u128]	[ValueType::Uint(scale)];
		[i128_value]	[i128]	[ValueType::Int(scale)];
	)]
	/// Returns the integer value of the bytes of this scalar
	///
	/// Returns none if not a value
	pub fn name(&self) -> Option<typ>
	{
		match self.extend(size_of::<typ>(), &value_typ([4]))
		{
			Scalar::Val(bytes) =>
			{
				paste::paste! {
					Some(LittleEndian::[<read_ typ >](&*bytes))
				}
			},
			_ => None,
		}
	}
}

/// A value is a (potential) vector of 0 or more scalars of the same type.
///
/// Values are consumed by instructions to perform actions or arithmetic.
///
/// Has a specific type, which dictates the type of all scalar elements.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Value
{
	/// Scalar type of the value
	pub typ: ValueType,

	/// First scalar (index 0)
	pub first: Scalar,

	/// Remaining scalars (index 1 and above)
	pub rest: Box<[Scalar]>,
}
impl Value
{
	/// Constructs a value with only the given scalar.
	pub fn singleton_typed(typ: ValueType, first: Scalar) -> Self
	{
		Self {
			typ,
			first,
			rest: Vec::from([]).into_boxed_slice(),
		}
	}

	/// Constructs a value with only the given scalar.
	///
	/// The type of value is equivalent to the given type parameter.
	pub fn singleton<N: PrimInt>(first: Scalar) -> Self
	{
		Self::singleton_typed(ValueType::new::<N>(), first)
	}

	/// Constructs a value with only one Nan scalar.
	///
	/// The type of value is equivalent to the given type parameter.
	pub fn new_nan<N: PrimInt>() -> Self
	{
		Self::singleton::<N>(Scalar::Nan)
	}

	/// Constructs a value with only one Nar scalar with the given payload.
	///
	/// The type of value is equivalent to the given type parameter.
	pub fn new_nar<N: PrimInt>(payload: usize) -> Self
	{
		Self::singleton::<N>(Scalar::Nar(payload))
	}

	/// Constructs a value with only one Nan scalar of the given type.
	pub fn new_nan_typed(typ: ValueType) -> Self
	{
		Self::singleton_typed(typ, Scalar::Nan)
	}

	/// Constructs a value with only one Nar scalar of the given type with the
	/// given payload.
	pub fn new_nar_typed(typ: ValueType, payload: usize) -> Self
	{
		Self::singleton_typed(typ, Scalar::Nar(payload))
	}

	/// Constructs a value with the given scalars of the given type.
	pub fn new_typed(typ: ValueType, first: Scalar, rest: Vec<Scalar>) -> Result<Self, ()>
	{
		if once(&first).chain(rest.iter()).all(|s| {
			match s
			{
				Scalar::Val(bytes) => bytes.len() == typ.scale(),
				_ => true,
			}
		})
		{
			Ok(Self {
				typ,
				first,
				rest: rest.into_boxed_slice(),
			})
		}
		else
		{
			Err(())
		}
	}

	/// Constructs a value with the given scalars.
	///
	/// The type of value is equivalent to the given type parameter.
	pub fn new<N: PrimInt>(first: Scalar, rest: Vec<Scalar>) -> Result<Self, ()>
	{
		Self::new_typed(ValueType::new::<N>(), first, rest)
	}

	/// Returns the type of the value.
	pub fn value_type(&self) -> ValueType
	{
		self.typ
	}

	/// Returns the number of bytes one scalar element if this value's type
	/// takes up.
	pub fn scale(&self) -> usize
	{
		self.typ.scale()
	}

	/// Returns the vector length of this value, i.e. the number of scalars in
	/// it.
	pub fn len(&self) -> usize
	{
		1 + self.rest.len()
	}

	/// Returns the number of bytes this value takes up in total.
	pub fn size(&self) -> usize
	{
		self.scale() * self.len()
	}

	/// Returns references to the scalars of this value in order
	pub fn iter(&self) -> impl Iterator<Item = &Scalar>
	{
		std::iter::once(&self.first).chain(self.rest.iter())
	}

	/// Returns mutable reference to the scalars of this value.
	///
	/// The caller must ensure only valid changes are made to the scalars.
	pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Scalar>
	{
		let first = &mut self.first;
		std::iter::once(first).chain(self.rest.iter_mut())
	}

	/// Returns a reference to the first scalar
	pub fn get_first(&self) -> &Scalar
	{
		&self.first
	}

	/// Returns a shared reference to the first scalar
	pub fn get_first_mut(&mut self) -> &mut Scalar
	{
		&mut self.first
	}
}
impl<N: PrimInt> From<N> for Value
{
	fn from(num: N) -> Self
	{
		let le = num.to_le();
		let bytes = unsafe {
			std::slice::from_raw_parts(
				std::mem::transmute::<&N, *const u8>(&le),
				std::mem::size_of::<N>(),
			)
		};
		Self::singleton::<N>(Scalar::Val(Box::from(bytes)))
	}
}
