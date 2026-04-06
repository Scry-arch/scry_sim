use quickcheck_macros::quickcheck;
use scry_sim::{HostMemory, Memory};

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
