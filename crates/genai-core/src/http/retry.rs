use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::time::Duration;

/// How often and how long to wait before retrying a failure that may pass.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct RetryPolicy {
    /// Retries after the first attempt. Two by default, like the providers' own SDKs.
    pub max_retries: u32,
    /// The wait before the first retry; it doubles each time.
    pub base_delay: Duration,
    /// The longest wait, including a provider's `retry-after`.
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        RetryPolicy { max_retries: 2, base_delay: Duration::from_millis(500), max_delay: Duration::from_secs(60) }
    }
}

impl RetryPolicy {
    /// Up to `max_retries` retries, waiting `base_delay` before the first and doubling up to `max_delay`.
    pub fn new(max_retries: u32, base_delay: Duration, max_delay: Duration) -> Self {
        RetryPolicy { max_retries, base_delay, max_delay }
    }

    /// No retries.
    pub fn none() -> Self {
        RetryPolicy { max_retries: 0, ..RetryPolicy::default() }
    }

    /// The wait before retry number `attempt` (from 0): exponential, capped, with up to 25% jitter taken off so clients
    /// that failed together don't retry together.
    pub fn delay(&self, attempt: u32) -> Duration {
        let full = self.base_delay.saturating_mul(1u32.checked_shl(attempt).unwrap_or(u32::MAX)).min(self.max_delay);
        let jitter = (random() % 250) as u32;
        full.saturating_sub(full * jitter / 1000)
    }
}

/// A random number, without a dependency: std seeds each `RandomState` randomly.
fn random() -> u64 {
    RandomState::new().build_hasher().finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backs_off_exponentially_with_a_cap() {
        let policy = RetryPolicy { max_retries: 5, base_delay: Duration::from_millis(100), max_delay: Duration::from_secs(1) };
        let first = policy.delay(0);
        assert!(first <= Duration::from_millis(100) && first >= Duration::from_millis(75));
        let third = policy.delay(2);
        assert!(third <= Duration::from_millis(400) && third >= Duration::from_millis(300));
        assert!(policy.delay(30) <= Duration::from_secs(1));
    }
}
