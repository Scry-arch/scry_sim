use duplicate::duplicate_item;
use quickcheck::{Arbitrary, Gen};
use quickcheck_macros::quickcheck;
use scry_sim::{
	arbitrary::{LimitedOps, NoCF, NoReads, Restriction, SimpleOps},
	CallFrameState, ExecState,
};
use std::{fmt::Debug, iter::empty};

/// Forces testing to not try to shrink the test arguments.
#[derive(Clone, Debug)]
struct NoShrink<T: Arbitrary + Debug + Clone>(T);

impl<T: Arbitrary + Debug + Clone> Arbitrary for NoShrink<T>
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		Self(T::arbitrary(g))
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		Box::new(empty())
	}
}

/// Tests that all shrinkage of call frames are valid
#[quickcheck]
#[ignore]
fn frame_shrinks_valid(frame: NoShrink<CallFrameState>) -> bool
{
	frame.0.shrink().all(|f| f.valid())
}

/// Tests that all shrinkage of states are valid
#[quickcheck]
#[ignore]
fn state_shrinks_valid(state: NoShrink<ExecState>) -> bool
{
	state.0.shrink().all(|f| f.validate().is_ok())
}

/// Test that all arbitrary execution state restricters shrink to valid state
/// that uphold the restriction
#[duplicate_item(
	restricter 		generics	test_name;
	[NoCF] 			[]			[shrink_no_cf];
	[NoReads] 		[]			[shrink_no_reads];
	[LimitedOps] 	[2,2]		[shrink_limited_ops_2_2];
	[SimpleOps] 	[]			[shrink_simple_ops];
)]
#[quickcheck]
#[ignore]
fn test_name(state: NoShrink<restricter<ExecState, generics>>) -> bool
{
	state.0.shrink().all(|state| {
		state.as_ref().validate().is_ok() && restricter::<_, generics>::restriction_holds(&state)
	})
}

#[quickcheck]
#[ignore]
fn shrink_no_cf_simple_ops_no_reads_limited_ops_2_2(
	state: NoShrink<NoCF<SimpleOps<NoReads<LimitedOps<ExecState, 2, 2>>>>>,
) -> bool
{
	state.0.shrink().all(|state| {
		state.as_ref().validate().is_ok()
			&& NoCF::restriction_holds(&state)
			&& SimpleOps::restriction_holds(&state)
			&& NoReads::restriction_holds(&state)
			&& LimitedOps::<_, 2, 2>::restriction_holds(&state)
	})
}

#[quickcheck]
#[ignore]
fn shrink_no_cf_limited_ops_0_0(state: NoShrink<NoCF<LimitedOps<ExecState, 0, 0>>>) -> bool
{
	state.0.shrink().all(|state| {
		state.as_ref().validate().is_ok()
			&& NoCF::restriction_holds(&state)
			&& LimitedOps::<_, 0, 0>::restriction_holds(&state)
	})
}
