use crate::value::{Scalar, Value, ValueType};
use byteorder::{ByteOrder, LittleEndian};
use quickcheck::{Arbitrary, Gen};

impl Arbitrary for ValueType
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		let signed = bool::arbitrary(g);
		let scale = u8::arbitrary(g) % 4;

		if signed
		{
			match scale
			{
				0 => Self::new::<i8>(),
				1 => Self::new::<i16>(),
				2 => Self::new::<i32>(),
				3 => Self::new::<i64>(),
				_ => unreachable!(),
			}
		}
		else
		{
			match scale
			{
				0 => Self::new::<u8>(),
				1 => Self::new::<u16>(),
				2 => Self::new::<u32>(),
				3 => Self::new::<u64>(),
				_ => unreachable!(),
			}
		}
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		match self
		{
			Self::Uint(v) if *v != 0 => Box::new(std::iter::once(Self::Uint(v / 2))),
			Self::Int(v) =>
			{
				let mut result = Vec::new();
				result.push(Self::Uint(*v));
				if *v != 0
				{
					result.push(Self::Int(*v / 2));
				}
				Box::new(result.into_iter())
			},
			_ => Box::new(std::iter::empty()),
		}
	}
}

/// Used to generate arbitrary scalars that are guaranteed to not be Nar/Nan and
/// have the given scale (as a power of 2).
///
/// Will at most generate 64-bit scalars
#[derive(Debug, Clone)]
pub struct ArbScalarVal(pub u8, pub Scalar);

impl Arbitrary for ArbScalarVal
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		let size_pow2 = u8::arbitrary(g) % 4;

		Self(
			size_pow2,
			Scalar::Val(match size_pow2
			{
				0 => Box::new(u8::arbitrary(g).to_le_bytes()),
				1 => Box::new(u16::arbitrary(g).to_le_bytes()),
				2 => Box::new(u32::arbitrary(g).to_le_bytes()),
				3 => Box::new(u64::arbitrary(g).to_le_bytes()),
				_ => unreachable!(),
			}),
		)
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		let mut result = Vec::new();

		let bytes = &self.1.bytes().unwrap();

		match self.0
		{
			0 =>
			{
				result.extend(
					bytes[0]
						.shrink()
						.map(|x| Self(self.0, Scalar::Val(Box::new(x.to_le_bytes())))),
				)
			},
			1 =>
			{
				result.extend(
					LittleEndian::read_u16(&bytes)
						.shrink()
						.map(|x| Self(self.0, Scalar::Val(Box::new(x.to_le_bytes())))),
				)
			},
			2 =>
			{
				result.extend(
					LittleEndian::read_u32(&bytes)
						.shrink()
						.map(|x| Self(self.0, Scalar::Val(Box::new(x.to_le_bytes())))),
				)
			},
			3 =>
			{
				result.extend(
					LittleEndian::read_u64(&bytes)
						.shrink()
						.map(|x| Self(self.0, Scalar::Val(Box::new(x.to_le_bytes())))),
				)
			},
			_ => unreachable!(),
		}

		Box::new(result.into_iter())
	}
}

/// Used to generate arbitrary values.
///
/// Can specify whether the generated values are allowed to be NaRs or NaNs
#[derive(Debug, Clone)]
pub struct ArbValue<const ALLOW_NAR: bool, const ALLOW_NAN: bool>(pub Value);
impl<const ALLOW_NAR: bool, const ALLOW_NAN: bool> Arbitrary for ArbValue<ALLOW_NAR, ALLOW_NAN>
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		let signed = bool::arbitrary(g);
		let ArbScalarVal(size_pow2, scalar) = ArbScalarVal::arbitrary(g);

		let typ = if signed
		{
			ValueType::Int(size_pow2)
		}
		else
		{
			ValueType::Uint(size_pow2)
		};
		// Add small chance of producing NaN and NaR
		Self(match u8::arbitrary(g)
		{
			0 if ALLOW_NAR => Value::new_nar_typed(typ, usize::arbitrary(g)),
			1 if ALLOW_NAN => Value::new_nan_typed(typ),
			_ => Value::singleton_typed(typ, scalar),
		})
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		let mut result = Vec::new();
		let size_pow2 = match self.0.value_type()
		{
			ValueType::Int(x) | ValueType::Uint(x) => x,
		};
		match self.0.iter().next().unwrap()
		{
			Scalar::Val(bytes) =>
			{
				if bytes.iter().all(|b| *b == 0) && ALLOW_NAN
				{
					// Shrink zeros to NaN
					result.push(Self(Value::new_nan_typed(self.0.value_type())));
				}
				else
				{
					result.extend(
						ArbScalarVal(size_pow2, Scalar::Val(bytes.clone()))
							.shrink()
							.map(|sc| Self(Value::singleton_typed(self.0.value_type(), sc.1))),
					);
				}
			},
			nan_or_nar =>
			{
				// Shrink type
				result.extend(self.0.value_type().shrink().map(|new_typ| {
					Self(match nan_or_nar
					{
						Scalar::Nan => Value::new_nan_typed(new_typ),
						Scalar::Nar(x) => Value::new_nar_typed(new_typ, *x),
						_ => unreachable!(),
					})
				}));
			},
		}
		Box::new(result.into_iter())
	}
}

impl Arbitrary for Value
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		ArbValue::<true, true>::arbitrary(g).0
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		Box::new(ArbValue::<true, true>(self.clone()).shrink().map(|v| v.0))
	}
}
