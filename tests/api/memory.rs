use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_sim::{BlockedMemory, HostMemory, Memory};

/// Tests HostMemory raw read
#[quickcheck]
fn host_read(data: u8) -> bool
{
	let buff = [data];

	let mut hm = HostMemory();

	data == hm.read_raw(buff.as_ptr() as usize).unwrap()
}

/// Tests HostMemory raw write
#[quickcheck]
fn host_write(data: u8) -> bool
{
	let buff = [0u8; 1];

	let mut hm = HostMemory();

	hm.write_raw(buff.as_ptr() as usize, data).unwrap();

	data == buff[0]
}

#[quickcheck]
fn blocked_memory_add_block_zeroed(offset: usize, size: usize, read_from: usize) -> TestResult
{
	if size.checked_add(1).is_none()
	{
		return TestResult::discard();
	}
	let size = (size % 1000) + 1; // No need for huge allocations
	if offset.checked_add(size).is_none()
	{
		return TestResult::discard();
	}

	let mut mem = BlockedMemory::empty();

	mem.add_block_zeroed(offset, size);

	TestResult::from_bool(mem.read_raw(offset + (read_from % size)).is_ok())
}
