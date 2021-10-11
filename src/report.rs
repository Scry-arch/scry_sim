/// A component that provides a report after each step.
///
/// Each step takes some arguments (who themselves could also provide reports)
/// and returns the report.
/// When another step is needed, the report can be reset to get the reporter
/// which can then take another step.
pub trait Reporter
{
	type Args;
	type Report: Report;

	fn step(self, args: Self::Args) -> Self::Report;
}

pub trait Report
{
	type Reporter: Reporter;

	fn reset(self) -> Self::Reporter;
}

impl Reporter for ()
{
	type Args = ();
	type Report = ();

	fn step(self, _: Self::Args) -> Self::Report
	{
		()
	}
}

impl Report for ()
{
	type Reporter = ();

	fn reset(self) -> Self::Reporter
	{
		()
	}
}

impl<A: Reporter, B: Reporter> Reporter for (A, B)
{
	type Args = (A::Args, B::Args);
	type Report = (A::Report, B::Report);

	fn step(self, (a, b): Self::Args) -> Self::Report
	{
		(self.0.step(a), self.1.step(b))
	}
}

impl<A: Report, B: Report> Report for (A, B)
{
	type Reporter = (A::Reporter, B::Reporter);

	fn reset(self) -> Self::Reporter
	{
		(self.0.reset(), self.1.reset())
	}
}

impl<A: Reporter, B: Reporter, C: Reporter> Reporter for (A, B, C)
{
	type Args = (A::Args, B::Args, C::Args);
	type Report = (A::Report, B::Report, C::Report);

	fn step(self, (a, b, c): Self::Args) -> Self::Report
	{
		(self.0.step(a), self.1.step(b), self.2.step(c))
	}
}

impl<A: Report, B: Report, C: Report> Report for (A, B, C)
{
	type Reporter = (A::Reporter, B::Reporter, C::Reporter);

	fn reset(self) -> Self::Reporter
	{
		(self.0.reset(), self.1.reset(), self.2.reset())
	}
}

impl<A: Reporter, B: Reporter, C: Reporter, D: Reporter> Reporter for (A, B, C, D)
{
	type Args = (A::Args, B::Args, C::Args, D::Args);
	type Report = (A::Report, B::Report, C::Report, D::Report);

	fn step(self, (a, b, c, d): Self::Args) -> Self::Report
	{
		(
			self.0.step(a),
			self.1.step(b),
			self.2.step(c),
			self.3.step(d),
		)
	}
}

impl<A: Report, B: Report, C: Report, D: Report> Report for (A, B, C, D)
{
	type Reporter = (A::Reporter, B::Reporter, C::Reporter, D::Reporter);

	fn reset(self) -> Self::Reporter
	{
		(
			self.0.reset(),
			self.1.reset(),
			self.2.reset(),
			self.3.reset(),
		)
	}
}

// let (exec_rep, mem_rep, data_rep) = (exec, mem, data).step();
// let (exec, mem, data) = (exec_rep, mem_rep, data_rep).reset();
