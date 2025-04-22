use quickcheck::TestResult;
use quickcheck_macros::quickcheck;
use scry_sim::{Metric, MetricReporter, MetricTracker, TrackReport};

#[quickcheck]
fn zero_metric_equals_absent(mut report: TrackReport, metric: Metric) -> TestResult
{
	report.reset_stat(metric);

	let mut report2 = report.clone();
	report2.add_stat(metric, 0);

	TestResult::from_bool(report == report2 && report2 == report)
}

#[quickcheck]
fn cloned_equal(report: TrackReport) -> TestResult
{
	let report2 = report.clone();

	TestResult::from_bool(report == report2 && report2 == report)
}

#[quickcheck]
fn added_metric_not_equal(report: TrackReport, metric: Metric, add: usize) -> TestResult
{
	if report
		.get_stat(metric)
		.checked_add(add)
		.and_then(|added| added.checked_add(1))
		.is_none()
	{
		return TestResult::discard();
	}

	let mut report2 = report.clone();
	report2.add_stat(metric, add + 1);

	TestResult::from_bool(report != report2 && report2 != report)
}
