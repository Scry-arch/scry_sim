use quickcheck::TestResult;
use scryer::{arbitrary::InstrAddr, CallFrameState, ControlFlowType, OperandState, ValueType};

/// Tests that all arbitrarily generated call frames are valid
#[quickcheck]
fn arb_valid(frame: CallFrameState) -> bool
{
	frame.valid()
}

/// Tests that any frame with the return address not 2-byte aligned is not
/// consistent
#[quickcheck]
fn unaligned_return_invalid(mut frame: CallFrameState) -> bool
{
	// Assume address is already aligned, unalign it.
	frame.ret_addr += 1;
	!frame.valid()
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
	!frame.valid()
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
	TestResult::from_bool(!frame.valid())
}

/// Tests that any frame with an operand referencing a non-existent read is
/// invalid
#[quickcheck]
fn reference_non_read_invalid(
	mut frame: CallFrameState,
	q_idx: usize,
	op_idx: usize,
	read_ref: usize,
) -> bool
{
	let op_q_len = frame.op_queues.len();
	let new_op = OperandState::MustRead(frame.reads.len().saturating_add(read_ref));

	if op_q_len == 0
	{
		frame.op_queues.insert(q_idx, (new_op, vec![]));
	}
	else
	{
		let (op1, op_rest) = frame.op_queues.iter_mut().nth(q_idx % op_q_len).unwrap().1;

		// Insert operand in appropriate place
		if op_idx % 4 == 0
		{
			let old_op1 = std::mem::replace(op1, new_op);
			op_rest.insert(0, old_op1);
		}
		else
		{
			op_rest.insert(op_idx % op_rest.len(), new_op);
		}
		// Ensure no reads are no longer referenced
		if op_rest.len() > 3
		{
			op_rest.pop().unwrap();
			frame.clean_reads();
		}
	}

	!frame.valid()
}

/// Tests that any frame with reads that aren't referenced by a MustRead operand
/// is invalid
#[quickcheck]
fn read_without_ref_invalid(
	mut frame: CallFrameState,
	read: (usize, usize, ValueType),
	extra_reads: Vec<(usize, usize, ValueType)>,
) -> bool
{
	// Add reads that can't possibly have references to them
	frame.reads.push(read);
	frame.reads.extend(extra_reads.into_iter());

	!frame.valid()
}

/// Tests that any frame with an operand queue of more than 4 operands is
/// invalid
#[quickcheck]
fn long_operand_queue_invalid(mut frame: CallFrameState, q_idx: usize) -> TestResult
{
	let op_qs_len = frame.op_queues.len();
	if op_qs_len == 0
	{
		return TestResult::discard();
	}
	let (_, (op1, op_rest)) = frame.op_queues.iter_mut().nth(q_idx % op_qs_len).unwrap();
	// Add 4 copies of the first operand, which results in at least 5 operands
	for _ in 0..4
	{
		op_rest.push(op1.clone());
	}
	TestResult::from_bool(!frame.valid())
}
