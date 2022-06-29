mod executor;

use scryasm::{Assemble, Raw};
use scryer::{
	execution::{ExecResult, Executor},
	memory::Memory,
	ExecState, Value,
};

#[macro_use(quickcheck)]
extern crate quickcheck_macros;

/// Tests that the given assembly program with the given inputs
/// returns the given outputs within the given execution steps
fn test_raw_program<'a, I: Iterator<Item = &'a str> + Clone>(
	asm: I,
	inputs: impl IntoIterator<Item = Value>,
	expected_outputs: impl IntoIterator<Item = Value>,
	max_execs: usize,
) -> Result<(), String>
{
	let object = Raw::assemble(asm).unwrap();
	let mem = Memory::new(object, 0);
	let mut exec = Executor::new(0, mem, inputs.into_iter());
	let count = 0;
	while count <= max_execs
	{
		match exec.step()
		{
			ExecResult::Ok(x) => exec = x,
			ExecResult::Done(result) =>
			{
				let expected = expected_outputs.into_iter().collect::<Vec<_>>();
				let actual = result.into_iter().collect::<Vec<_>>();

				if expected == actual
				{
					return Ok(());
				}
				else
				{
					return Err(format!("{:?} != {:?}", expected, actual));
				}
			},
			_ => return Err("Unexpected execution".into()),
		}
	}
	Err("Timeout".into())
}

macro_rules! test_program {
	(
		($timeout:expr) $inputs:expr => $outputs:expr;
		$($asm:tt)*
	) => {
		let result = test_raw_program(std::iter::once(stringify!($($asm)*)), $inputs, $outputs, $timeout);

		match result {
			Ok(_) => true,
			Err(e) => {
				println!("{}", e);
				false
			}
		}
	};
}

/// Tests that a function comprising just an immediate return returns nothing
/// regardless of inputs.
#[quickcheck]
fn empty_function_returns_nothing(inputs: Vec<Value>) -> bool
{
	test_program! {
		(2) inputs => [];
					ret ret_loc
		ret_loc:
	}
}

/// Tests that a function comprising just an immediate return returns nothing
/// regardless of inputs.
///
/// Tested for up to 4 inputs
#[quickcheck]
fn identity_function_returns_inputs(
	inputs: (Option<Value>, Option<Value>, Option<Value>, Option<Value>),
) -> bool
{
	let inputs = inputs
		.0
		.into_iter()
		.chain(inputs.1.into_iter())
		.chain(inputs.2.into_iter())
		.chain(inputs.3.into_iter());
	test_program! {
		(2) inputs.clone() => inputs;
					echo =>ret_loc
					ret ret_loc
		ret_loc:
	}
}
