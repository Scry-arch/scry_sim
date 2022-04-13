use scryasm::{Assemble, Raw};
use scryer::{
	data::Value,
	execution::{ExecResult, Executor},
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
) -> Result<(), &'static str>
{
	let object = Raw::assemble(asm).unwrap();
	let mem = scryer::memory::Memory::new(object, 0);
	let mut exec = Executor::new(0, mem, inputs.into_iter());
	let count = 0;
	while count <= max_execs
	{
		match exec.execute()
		{
			ExecResult::Ok(x) => exec = x,
			ExecResult::Done(mut result) =>
			{
				let expected = expected_outputs.into_iter().collect::<Vec<_>>();
				let actual = result.into_iter().collect::<Vec<_>>();
				assert_eq!(expected, actual);
				return Ok(());
			},
			_ => return Err("Unexpected execution"),
		}
	}
	Err("Timeout")
}

macro_rules! test_program {
	(
		($timeout:expr) $inputs:expr => $outputs:expr;
		$asm:literal
	) => {
		test_raw_program(std::iter::once($asm), $inputs, $outputs, $timeout).is_ok()
	};
}

/// Tests that a function comprising just an immediate return returns the inputs
/// given to it.
///
/// Tests up to 4 inputs
#[quickcheck]
fn empty_function_returns_inputs(
	input: (Option<Value>, Option<Value>, Option<Value>, Option<Value>),
) -> bool
{
	let inputs = input
		.0
		.into_iter()
		.chain(input.1.into_iter())
		.chain(input.2.into_iter())
		.chain(input.3.into_iter());
	test_program! {
		(2) inputs.clone() => inputs;
		"ret 0"
	}
}
