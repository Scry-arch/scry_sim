use duplicate::duplicate_item;
use quickcheck_macros::quickcheck;
use scry_sim::{
	arbitrary::{LimitedOps, NoCF, NoReads, Restriction, SimpleOps},
	CallFrameState, ExecState, ValueType,
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
	idx: usize,
	read: (bool, usize, usize, ValueType),
) -> bool
{
	// Create invalid by adding a superfluous read
	frame.reads.push(read);

	// Insert frame into state
	let insert_idx = idx % (state.frame_stack.len() + 1);
	if insert_idx == 0
	{
		let old = std::mem::replace(&mut state.frame, frame);
		state.frame_stack.insert(0, old);
	}
	else
	{
		state.frame_stack.insert(insert_idx - 1, frame);
	}

	state.validate().is_err()
}

/// Test that all arbitrary execution state restricters produce valid states
/// that also uphold their own restrictions
#[duplicate_item(
	restricter 	test_name;
	[NoCF] 		[arb_state_no_cf];
	[NoReads] 	[arb_state_no_reads];
)]
#[quickcheck]
fn test_name(state: restricter<ExecState>) -> bool
{
	state.as_ref().validate().is_ok() && restricter::restriction_holds(&state)
}

#[quickcheck]
fn no_cf_simple_ops_no_reads_limited_ops_2_2(
	state: NoCF<SimpleOps<NoReads<LimitedOps<ExecState, 2, 2>>>>,
) -> bool
{
	NoCF::restriction_holds(&state)
		&& SimpleOps::restriction_holds(&state)
		&& NoReads::restriction_holds(&state)
		&& LimitedOps::<_, 2, 2>::restriction_holds(&state)
}
