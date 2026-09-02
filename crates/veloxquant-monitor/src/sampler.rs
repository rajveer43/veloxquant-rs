//! A broadcast-based metrics sampler.
//!
//! **v0.1.0 status:** [`Monitor`] provides the subscription plumbing
//! (`start`/`subscribe`) and lets callers publish samples via
//! [`Monitor::publish`]. Automatic periodic sampling of live memory/KV
//! metrics (wiring this up to `veloxquant-system` and a running inference
//! session) is planned for v0.3.0 alongside AutoPilot and benchmarking.

use tokio::sync::broadcast;

use crate::metrics::Metrics;

const DEFAULT_CHANNEL_CAPACITY: usize = 64;

/// Publishes [`Metrics`] samples to any number of subscribers.
///
/// Backed by [`tokio::sync::broadcast`], so subscribers that fall behind
/// drop old samples rather than blocking the publisher, and disconnecting
/// is as simple as dropping the receiver.
#[derive(Debug, Clone)]
pub struct Monitor {
    sender: broadcast::Sender<Metrics>,
}

impl Default for Monitor {
    fn default() -> Self {
        Self::new()
    }
}

impl Monitor {
    /// Creates a new, unstarted monitor.
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(DEFAULT_CHANNEL_CAPACITY);
        Self { sender }
    }

    /// Marks the monitor as active. In v0.1.0 this is a no-op beyond
    /// confirming the channel is live, since automatic sampling isn't
    /// implemented yet; callers drive metrics via [`Monitor::publish`].
    pub async fn start(&self) -> veloxquant_core::Result<()> {
        Ok(())
    }

    /// Publishes a metrics sample to all current subscribers. Returns the
    /// number of subscribers the sample was delivered to; `0` if there are
    /// none currently subscribed (not an error).
    pub fn publish(&self, metrics: Metrics) -> usize {
        self.sender.send(metrics).unwrap_or(0)
    }

    /// Subscribes to future metrics samples.
    pub fn subscribe(&self) -> broadcast::Receiver<Metrics> {
        self.sender.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn subscriber_receives_published_metrics() {
        let monitor = Monitor::new();
        let mut receiver = monitor.subscribe();

        monitor.start().await.unwrap();
        let sent = Metrics {
            memory_used_bytes: 1024,
            ..Default::default()
        };
        monitor.publish(sent);

        let received = receiver.recv().await.unwrap();
        assert_eq!(received.memory_used_bytes, 1024);
    }

    #[tokio::test]
    async fn multiple_subscribers_each_receive_samples() {
        let monitor = Monitor::new();
        let mut a = monitor.subscribe();
        let mut b = monitor.subscribe();

        monitor.publish(Metrics::default());

        assert!(a.recv().await.is_ok());
        assert!(b.recv().await.is_ok());
    }

    #[test]
    fn publish_without_subscribers_returns_zero() {
        let monitor = Monitor::new();
        assert_eq!(monitor.publish(Metrics::default()), 0);
    }
}
