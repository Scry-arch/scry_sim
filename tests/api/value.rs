use duplicate::duplicate;
use scry_sim::ValueType;

/// Tests the `ValueType::get_effective_type` method.
#[test]
fn get_effective_type()
{
	duplicate! {
		[
			typ1	typ2	typ3;
			[u8] 	[u8] 	[u8];
			[u8] 	[i8] 	[u8];
			[i8] 	[u8] 	[u8];
			[i8] 	[i8] 	[i8];
			[u8] 	[u16]	[u16];
			[u32]	[u16]	[u32];
			[i8] 	[u16]	[u16];
			[u8] 	[i16]	[u16];
			[u64] 	[i16]	[u64];
			[i64] 	[i16]	[i64];
		]
		assert_eq!(ValueType::typ1().get_effective_type(&ValueType::typ2()), ValueType::typ3());
	}
}
