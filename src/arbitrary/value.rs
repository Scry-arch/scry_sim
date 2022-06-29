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

impl Arbitrary for Value
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		let signed = bool::arbitrary(g);
		let scale = u8::arbitrary(g) % 4;

		let value: Self = if signed
		{
			match scale
			{
				0 => i8::arbitrary(g).into(),
				1 => i16::arbitrary(g).into(),
				2 => i32::arbitrary(g).into(),
				3 => i64::arbitrary(g).into(),
				_ => unreachable!(),
			}
		}
		else
		{
			match scale
			{
				0 => u8::arbitrary(g).into(),
				1 => u16::arbitrary(g).into(),
				2 => u32::arbitrary(g).into(),
				3 => u64::arbitrary(g).into(),
				_ => unreachable!(),
			}
		};

		// Add small change of producing NaN and NaR
		match u8::arbitrary(g)
		{
			0 => Value::new_nar_typed(value.value_type(), usize::arbitrary(g)),
			1 => Value::new_nan_typed(value.value_type()),
			_ => value,
		}
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		let mut result = Vec::new();

		match self.iter().next().unwrap()
		{
			Scalar::Val(bytes) =>
			{
				if bytes.iter().all(|b| *b == 0)
				{
					// Shrink zeros to NaN
					result.push(Value::new_nan_typed(self.value_type()));
				}
				else
				{
					match self.value_type()
					{
						ValueType::Int(pow2_size) =>
						{
							match pow2_size
							{
								// Shrinkers for signed ints are bugged, so shrink manually
								// See: https://github.com/BurntSushi/quickcheck/issues/295#issuecomment-895491930
								0 => result.push(((bytes[0] as i8) / 2).into()),
								1 => result.push((LittleEndian::read_i16(&bytes) / 2).into()),
								2 => result.push((LittleEndian::read_i32(&bytes) / 2).into()),
								3 => result.push((LittleEndian::read_i64(&bytes) / 2).into()),
								_ => unreachable!(),
							}
						},
						ValueType::Uint(pow2_size) =>
						{
							match pow2_size
							{
								0 => result.extend(bytes[0].shrink().map(|x| x.into())),
								1 =>
								{
									result.extend(
										LittleEndian::read_u16(&bytes).shrink().map(|x| x.into()),
									)
								},
								2 =>
								{
									result.extend(
										LittleEndian::read_u32(&bytes).shrink().map(|x| x.into()),
									)
								},
								3 =>
								{
									result.extend(
										LittleEndian::read_u64(&bytes).shrink().map(|x| x.into()),
									)
								},
								_ => unreachable!(),
							}
						},
					}
				}
			},
			nan_or_nar =>
			{
				// Shrink type
				result.extend(self.value_type().shrink().map(|new_typ| {
					match nan_or_nar
					{
						Scalar::Nan => Value::new_nan_typed(new_typ),
						Scalar::Nar(x) => Value::new_nar_typed(new_typ, *x),
						_ => unreachable!(),
					}
				}));
			},
		}
		Box::new(result.into_iter())
	}
}
