use scryer::ExecState;
use scryer::execution::Executor;
use scryer::memory::Memory;

mod control_flow;

/// Tests can convert from state to executor back to state without the state
/// changing
#[quickcheck]
fn import_export_executor(state: ExecState, mem: Vec<u8>, base_addr: usize) -> bool
{
	let executor = Executor::from_state(&state, Memory::new(mem, base_addr));
	let new_state = executor.state();
	
	new_state == state
}