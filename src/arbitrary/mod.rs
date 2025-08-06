mod exec_state;
mod metrics;
mod value;

pub use exec_state::*;
use quickcheck::Gen;
pub use value::*;

/// Generate arbitrary values that depend on som 'weight'.
///
/// The meaning of the weight it implementation dependent.
/// It could typically be some restriction or requirement that the resulting
/// value must uphold.
///
/// The weight is passed to the functions must be the same.
pub trait WeightedArbitrary<W>: Clone + 'static
{
	fn arbitrary_weighted(g: &mut Gen, weight: W) -> Self;

	fn shrink_weighted(&self, _weight: W) -> Box<dyn Iterator<Item = Self>>;
}
