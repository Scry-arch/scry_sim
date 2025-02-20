use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_sim::{
	arbitrary::InstrAddr, CallFrameState, ControlFlowType, OperandList, OperandState, ValueType,
};

/// Tests that all arbitrarily generated call frames are valid
#[quickcheck]
fn arb_valid(frame: CallFrameState) -> bool
{
	frame.validate().is_ok()
}

/// Tests that any frame with the return address not 2-byte aligned is not
/// consistent
#[quickcheck]
fn unaligned_return_invalid(mut frame: CallFrameState) -> bool
{
	// Assume address is already aligned, unalign it.
	frame.ret_addr += 1;
	frame.validate().is_err()
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
	frame.validate().is_err()
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
	TestResult::from_bool(frame.validate().is_err())
}

/// Tests that any frame with an operand referencing a non-existent read is
/// invalid
#[quickcheck]
fn reference_non_read_invalid(
	mut frame: CallFrameState,
	list_idx: usize,
	op_idx: usize,
	read_ref: usize,
) -> bool
{
	let op_q_len = frame.op_queue.len();
	let new_op = OperandState::MustRead(frame.reads.len().saturating_add(read_ref));

	if op_q_len == 0
	{
		frame
			.op_queue
			.insert(list_idx, OperandList::new(new_op, vec![]));
	}
	else
	{
		let op_list = frame
			.op_queue
			.iter_mut()
			.nth(list_idx % op_q_len)
			.unwrap()
			.1;

		// Insert operand in appropriate place
		if op_idx % 4 == 0
		{
			let old_op1 = std::mem::replace(&mut op_list.first, new_op);
			op_list.rest.insert(0, old_op1);
		}
		else
		{
			op_list.rest.insert(op_idx % op_list.rest.len(), new_op);
		}
		// Ensure no reads are no longer referenced
		if op_list.len() > 4
		{
			op_list.rest.pop().unwrap();
			frame.clean_reads();
		}
	}

	frame.validate().is_err()
}

/// Tests that any frame with reads that aren't referenced by a MustRead operand
/// is invalid
#[quickcheck]
fn read_without_ref_invalid(
	mut frame: CallFrameState,
	read: (bool, usize, usize, ValueType),
	extra_reads: Vec<(bool, usize, usize, ValueType)>,
) -> bool
{
	// Add reads that can't possibly have references to them
	frame.reads.push(read);
	frame.reads.extend(extra_reads.into_iter());

	frame.validate().is_err()
}

/// Tests that any frame with an operand list of more than 4 operands is
/// invalid
#[quickcheck]
fn long_operand_list_invalid(mut frame: CallFrameState, list_idx: usize) -> TestResult
{
	let op_qs_len = frame.op_queue.len();
	if op_qs_len == 0
	{
		return TestResult::discard();
	}
	let (_, op_list) = frame.op_queue.iter_mut().nth(list_idx % op_qs_len).unwrap();
	// Add 4 copies of the first operand, which results in at least 5 operands
	for _ in 0..4
	{
		op_list.push(op_list.first.clone());
	}
	TestResult::from_bool(frame.validate().is_err())
}

/// Tests that any frame is equals to itself
#[quickcheck]
fn equal_self(frame: CallFrameState) -> bool
{
	frame == frame
}
