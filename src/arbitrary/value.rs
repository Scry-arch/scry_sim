use crate::data::{Value, ValueState, ValueType};
use byteorder::{ByteOrder, LittleEndian};
use quickcheck::{Arbitrary, Gen};

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
			ValueState::Val(bytes) =>
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
								0 => result.extend((bytes[0] as i8).shrink().map(|x| x.into())),
								1 =>
								{
									result.extend(
										LittleEndian::read_i16(&bytes).shrink().map(|x| x.into()),
									)
								},
								2 =>
								{
									result.extend(
										LittleEndian::read_i32(&bytes).shrink().map(|x| x.into()),
									)
								},
								3 =>
								{
									result.extend(
										LittleEndian::read_i64(&bytes).shrink().map(|x| x.into()),
									)
								},
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
			ValueState::Nan | ValueState::Nar(_) => (),
		}

		Box::new(result.into_iter())
	}
}
