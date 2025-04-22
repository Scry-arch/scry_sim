use crate::{Metric, MetricReporter, MetricTracker, TrackReport};
use quickcheck::{Arbitrary, Gen};
use strum::IntoEnumIterator;

impl Arbitrary for Metric
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		Metric::iter()
			.nth(usize::arbitrary(g) % Metric::iter().count())
			.unwrap()
	}
}

impl Arbitrary for TrackReport
{
	fn arbitrary(g: &mut Gen) -> Self
	{
		let mut report = TrackReport::new();

		Metric::iter().for_each(|m| {
			report.add_stat(m, Arbitrary::arbitrary(g));
		});

		report
	}

	fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
	{
		let clone = self.clone();
		Box::new(Metric::iter().flat_map(move |m| {
			let clone = clone.clone();
			let old_value = clone.get_stat(m);

			old_value.shrink().map(move |v| {
				let mut shrunk = clone.clone();
				shrunk.reset_stat(m);
				shrunk.add_stat(m, v);
				shrunk
			})
		}))
	}
}
