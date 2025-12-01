use duplicate::duplicate_item;
use quickcheck_macros::quickcheck;
use scry_sim::{
	arbitrary::{LimitedOps, NoCF, Restriction, SimpleOps},
	CallFrameState, CallType, ExecState, Value,
};

/// Tests that all arbitrarily generated states are valid
#[quickcheck]
fn arb_state_is_valid(state: ExecState) -> bool
{
	state.validate().is_ok()
}

/// Tests that any frame with the return address not 2-byte aligned is not
/// consistent
#[quickcheck]
fn state_unaligned_address(mut state: ExecState) -> bool
{
	// Assume address is already aligned, unalign it.
	state.address += 1;
	state.validate().is_err()
}

/// Tests that any state with an invalid frame is invalid
#[quickcheck]
fn state_invalid_frame(
	mut state: ExecState,
	mut frame: CallFrameState,
	call_type: CallType<Value>,
	idx: usize,
) -> bool
{
	// Create invalid by making return address unaligned
	if frame.ret_addr % 2 == 0
	{
		frame.ret_addr += 1;
	}

	// Insert frame into state
	let insert_idx = idx % (state.frame_stack.len() + 1);
	if insert_idx == 0
	{
		let old = std::mem::replace(&mut state.frame, frame);
		state.frame_stack.insert(0, (old, call_type));
	}
	else
	{
		state.frame_stack.insert(insert_idx - 1, (frame, call_type));
	}

	state.validate().is_err()
}

/// Test that all arbitrary execution state restricters produce valid states
/// that also uphold their own restrictions
#[duplicate_item(
	restricter 	test_name;
	[NoCF] 		[arb_state_no_cf];
)]
#[quickcheck]
fn test_name(state: restricter<ExecState>) -> bool
{
	state.as_ref().validate().is_ok() && restricter::restriction_holds(&state)
}

#[quickcheck]
fn no_cf_simple_ops_limited_ops_2_2(state: NoCF<SimpleOps<LimitedOps<ExecState, 2, 2>>>) -> bool
{
	NoCF::restriction_holds(&state)
		&& SimpleOps::restriction_holds(&state)
		&& LimitedOps::<_, 2, 2>::restriction_holds(&state)
}
