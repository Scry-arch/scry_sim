use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_sim::{arbitrary::InstrAddr, CallFrameState, ControlFlowType, MAX_INSTR_INPUTS};

/// Tests that all arbitrarily generated call frames are valid
#[quickcheck]
fn arb_valid(frame: CallFrameState) -> bool
{
	frame
		.validate(usize::MAX, usize::MAX - MAX_INSTR_INPUTS)
		.is_ok()
}

/// Tests that any frame with the return address not 2-byte aligned is not
/// consistent
#[quickcheck]
fn unaligned_return_invalid(mut frame: CallFrameState) -> bool
{
	// Assume address is already aligned, unalign it.
	frame.ret_addr += 1;
	frame.validate(usize::MAX, usize::MAX).is_err()
}

/// Tests that any frame with the return address not 2-byte aligned is invalid
#[quickcheck]
fn unaligned_control_trigger_invalid(
	mut frame: CallFrameState,
	InstrAddr(trigger): InstrAddr,
	typ: ControlFlowType,
) -> bool
{
	// Address is already aligned, unalign it.
	frame.branches.insert(trigger + 1, typ);
	frame.validate(usize::MAX, usize::MAX).is_err()
}

/// Tests that any frame with the control flow target address not 2-byte aligned
/// is invalid
#[quickcheck]
fn unaligned_control_target_invalid(
	mut frame: CallFrameState,
	InstrAddr(trigger): InstrAddr,
	typ: ControlFlowType,
) -> TestResult
{
	use ControlFlowType::*;
	// Address is already aligned, unalign it.
	frame.branches.insert(
		trigger,
		match typ
		{
			Branch(targ) => Branch(targ + 1),
			Call(targ) => Call(targ + 1),
			Return => return TestResult::discard(),
		},
	);
	TestResult::from_bool(frame.validate(usize::MAX, usize::MAX).is_err())
}

/// Tests that any frame with an operand list of more operands than the allowed
/// depth is invalid
#[quickcheck]
fn long_operand_list_invalid(
	mut frame: CallFrameState,
	list_idx: usize,
	more_depth: usize,
) -> TestResult
{
	let more_depth = more_depth % 16;
	let max_depth = more_depth + MAX_INSTR_INPUTS;
	let op_qs_len = frame.op_queue.len();
	if op_qs_len == 0
	{
		return TestResult::discard();
	}
	let (_, op_list) = frame.op_queue.iter_mut().nth(list_idx % op_qs_len).unwrap();
	// Add max_depth copies of the first operand, which results in at least
	// max_depth operands
	for _ in 0..max_depth
	{
		op_list.push(op_list.first.clone());
	}
	TestResult::from_bool(frame.validate(usize::MAX, more_depth).is_err())
}

/// Tests that any frame is equals to itself
#[quickcheck]
fn equal_self(frame: CallFrameState) -> bool
{
	frame == frame
}
