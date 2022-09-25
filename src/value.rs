use num_traits::PrimInt;
use std::iter::once;

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

	/// Returns the number of bytes one scalar element if this type takes up.
	pub fn scale(&self) -> usize
	{
		match self
		{
			ValueType::Uint(x) | ValueType::Int(x) => 2usize.pow(*x as u32),
		}
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
	pub(crate) fn set_val(&mut self, bytes: &[u8])
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
	typ: ValueType,

	/// First scalar (index 0)
	first: Scalar,

	/// Remaining scalars (index 1 and above)
	rest: Box<[Scalar]>,
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
