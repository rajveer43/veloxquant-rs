//! A broadcast-based metrics monitor, plus a periodic sampler that drives
//! it.
//!
//! [`Monitor`] is the pub/sub plumbing: [`Monitor::publish`] fans a
//! [`Metrics`] sample out to every [`Monitor::subscribe`]r over a
//! [`tokio::sync::broadcast`] channel. [`Monitor::spawn_sampler`] adds the
//! periodic half — a background task that polls a [`Sampler`] on an
//! interval and publishes each sample through that same channel (it calls
//! [`Monitor::publish`]; there is no second channel). The two compose:
//! out-of-band events (e.g. a finished inference request) can still be
//! published by hand between ticks, like Go's `Monitor.Report`.
//!
//! [`SystemSampler`] is the default [`Sampler`]: host memory used/available
//! via `veloxquant-system`, matching the sampler Go's `Client.Monitor`
//! wires up. The inference-side fields of [`Metrics`] (tokens/sec, TTFT,
//! KV-cache bytes, compression ratio) stay at their defaults in a
//! `SystemSampler` sample — nothing on the host can observe them without a
//! running inference session, so they are left for callers to publish (or
//! for a custom [`Sampler`] to fill in).

use std::time::Duration;

use tokio::sync::{broadcast, watch};
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;

use crate::metrics::Metrics;

const DEFAULT_CHANNEL_CAPACITY: usize = 64;

/// The sampling interval used when [`Monitor::spawn_sampler`] is given a
/// zero interval. Matches Go's `monitor.New` default (5 s).
pub const DEFAULT_SAMPLE_INTERVAL: Duration = Duration::from_secs(5);

/// Produces a [`Metrics`] snapshot on demand — Go's `monitor.Sampler`.
///
/// Implemented with `#[async_trait::async_trait]` (like the facade crate's
/// `Tool` trait) rather than a native `async fn` in the trait, because the
/// sampling loop runs on a spawned task and so needs the returned future to
/// be `Send`, which a native `async fn` in a trait cannot promise at this
/// workspace's MSRV (1.75).
#[async_trait::async_trait]
pub trait Sampler: Send + Sync + 'static {
    /// Takes one sample. An `Err` skips this tick (nothing is published)
    /// without stopping the sampling loop — Go's `sampleAndNotify`
    /// behavior.
    async fn sample(&self) -> veloxquant_core::Result<Metrics>;
}

/// Samples host memory via [`veloxquant_system::memory_stats`]: fills
/// `memory_used_bytes` (`total - available`) and `memory_available_bytes`,
/// leaving every inference-side field at its default. Mirrors the sampler
/// Go's `Client.Monitor` constructs.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemSampler;

impl SystemSampler {
    /// Creates a new [`SystemSampler`].
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Sampler for SystemSampler {
    async fn sample(&self) -> veloxquant_core::Result<Metrics> {
        // `memory_stats` is a single, fast `sysinfo` refresh; it is called
        // inline rather than via `spawn_blocking` to avoid requiring a
        // multi-threaded runtime.
        Ok(metrics_from_memory_stats(veloxquant_system::memory_stats()))
    }
}

/// The [`Metrics`] a [`SystemSampler`] reports for a memory snapshot:
/// `used = total - available` (Go's `TotalMemory - AvailableMemory`),
/// inference-side fields left at their defaults.
///
/// Values are passed through as `veloxquant-system` reports them. On macOS
/// that is `sysinfo`'s "available" figure, which subtracts compressed-memory
/// pages and so can read `0` on a Mac with a large compressor pool even
/// when plenty of memory is reclaimable.
pub fn metrics_from_memory_stats(stats: veloxquant_system::MemoryStats) -> Metrics {
    Metrics {
        memory_used_bytes: stats.total_bytes.saturating_sub(stats.available_bytes),
        memory_available_bytes: stats.available_bytes,
        ..Default::default()
    }
}

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

    /// Confirms the channel is live. Kept for source compatibility with
    /// v0.1.0; it does **not** start sampling (it has no way to return the
    /// handle that stops it). Use [`Monitor::spawn_sampler`] for periodic
    /// sampling.
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

    /// Starts a background task that calls `sampler` every `interval` and
    /// publishes each successful sample via [`Monitor::publish`].
    ///
    /// The first sample is taken immediately, then once per `interval`
    /// (Go's `Monitor.run`). A zero `interval` falls back to
    /// [`DEFAULT_SAMPLE_INTERVAL`] (Go's `monitor.New`). If a sample takes
    /// longer than `interval`, the next tick is delayed rather than fired
    /// in a burst to catch up. A sampler error skips that tick only.
    ///
    /// Sampling runs until [`SamplingHandle::stop`] is awaited or the
    /// handle is dropped. Several samplers may run against one monitor;
    /// their samples interleave on the same channel.
    ///
    /// Must be called from within a Tokio runtime (it uses
    /// [`tokio::spawn`]).
    pub fn spawn_sampler<S: Sampler>(&self, sampler: S, interval: Duration) -> SamplingHandle {
        let interval = if interval.is_zero() {
            DEFAULT_SAMPLE_INTERVAL
        } else {
            interval
        };
        let (stop_tx, mut stop_rx) = watch::channel(false);
        let monitor = self.clone();

        let task = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    biased;
                    // Fires on an explicit stop *and* when the handle (the
                    // only sender) is dropped.
                    _ = stop_rx.changed() => return,
                    _ = ticker.tick() => {
                        if let Ok(metrics) = sampler.sample().await {
                            monitor.publish(metrics);
                        }
                    }
                }
            }
        });

        SamplingHandle {
            stop: stop_tx,
            task: Some(task),
            interval,
        }
    }
}

/// Controls a sampling task started by [`Monitor::spawn_sampler`].
///
/// Dropping the handle signals the task to stop (without waiting for it);
/// [`SamplingHandle::stop`] signals and then waits, like Go's
/// `Monitor.Stop`.
#[derive(Debug)]
pub struct SamplingHandle {
    stop: watch::Sender<bool>,
    task: Option<JoinHandle<()>>,
    interval: Duration,
}

impl SamplingHandle {
    /// The effective sampling interval (after the zero-interval fallback).
    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Whether the sampling task has exited.
    pub fn is_finished(&self) -> bool {
        self.task.as_ref().is_none_or(JoinHandle::is_finished)
    }

    /// Stops sampling and waits for the background task to exit. A sample
    /// already in progress is allowed to finish first; nothing is
    /// published after this returns.
    pub async fn stop(mut self) {
        let _ = self.stop.send(true);
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }
}

impl Drop for SamplingHandle {
    fn drop(&mut self) {
        let _ = self.stop.send(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

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

    /// Counts calls and reports the call number as `memory_used_bytes`, so
    /// tests can check ordering and cadence. Fails every call listed in
    /// `fail_on`.
    struct CountingSampler {
        calls: Arc<AtomicUsize>,
        fail_on: Vec<usize>,
    }

    impl CountingSampler {
        fn new() -> (Self, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    calls: calls.clone(),
                    fail_on: Vec::new(),
                },
                calls,
            )
        }
    }

    #[async_trait::async_trait]
    impl Sampler for CountingSampler {
        async fn sample(&self) -> veloxquant_core::Result<Metrics> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if self.fail_on.contains(&n) {
                return Err(veloxquant_core::VeloxQuantError::InvalidRequest(format!(
                    "sample {n} failed"
                )));
            }
            Ok(Metrics {
                memory_used_bytes: n as u64,
                ..Default::default()
            })
        }
    }

    #[tokio::test(start_paused = true)]
    async fn sampler_publishes_immediately_then_once_per_interval() {
        let monitor = Monitor::new();
        let mut rx = monitor.subscribe();
        let (sampler, calls) = CountingSampler::new();

        let handle = monitor.spawn_sampler(sampler, Duration::from_secs(1));

        // First sample is immediate (Go samples before the first tick).
        assert_eq!(rx.recv().await.unwrap().memory_used_bytes, 1);

        // Then one per interval: with paused time, recv() auto-advances the
        // clock to the next tick, so samples 2 and 3 arrive in order.
        let before = tokio::time::Instant::now();
        assert_eq!(rx.recv().await.unwrap().memory_used_bytes, 2);
        assert_eq!(rx.recv().await.unwrap().memory_used_bytes, 3);
        assert_eq!(before.elapsed(), Duration::from_secs(2));

        handle.stop().await;
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test(start_paused = true)]
    async fn sampler_error_skips_tick_without_stopping_loop() {
        let monitor = Monitor::new();
        let mut rx = monitor.subscribe();
        let (mut sampler, _calls) = CountingSampler::new();
        sampler.fail_on = vec![2];

        let handle = monitor.spawn_sampler(sampler, Duration::from_millis(100));

        assert_eq!(rx.recv().await.unwrap().memory_used_bytes, 1);
        // Sample 2 errored: nothing was published for it, and sampling
        // carried on to sample 3.
        assert_eq!(rx.recv().await.unwrap().memory_used_bytes, 3);

        handle.stop().await;
    }

    #[tokio::test(start_paused = true)]
    async fn stop_halts_publishing() {
        let monitor = Monitor::new();
        let mut rx = monitor.subscribe();
        let (sampler, calls) = CountingSampler::new();

        let handle = monitor.spawn_sampler(sampler, Duration::from_millis(100));
        rx.recv().await.unwrap();
        handle.stop().await;

        let calls_at_stop = calls.load(Ordering::SeqCst);
        tokio::time::sleep(Duration::from_secs(10)).await;
        assert_eq!(calls.load(Ordering::SeqCst), calls_at_stop);
        assert!(matches!(
            rx.try_recv(),
            Err(broadcast::error::TryRecvError::Empty)
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn dropping_handle_stops_sampling() {
        let monitor = Monitor::new();
        let mut rx = monitor.subscribe();
        let (sampler, calls) = CountingSampler::new();

        let handle = monitor.spawn_sampler(sampler, Duration::from_millis(100));
        rx.recv().await.unwrap();
        drop(handle);

        // Let the task observe the drop, then run well past many ticks.
        tokio::time::sleep(Duration::from_secs(10)).await;
        let settled = calls.load(Ordering::SeqCst);
        tokio::time::sleep(Duration::from_secs(10)).await;
        assert_eq!(calls.load(Ordering::SeqCst), settled);
        assert!(settled <= 2, "sampling continued after drop: {settled}");
    }

    #[tokio::test(start_paused = true)]
    async fn zero_interval_falls_back_to_default() {
        let monitor = Monitor::new();
        let (sampler, _calls) = CountingSampler::new();
        let handle = monitor.spawn_sampler(sampler, Duration::ZERO);
        assert_eq!(handle.interval(), DEFAULT_SAMPLE_INTERVAL);
        assert!(!handle.is_finished());
        handle.stop().await;
    }

    #[tokio::test(start_paused = true)]
    async fn manual_publish_interleaves_with_sampled_metrics() {
        let monitor = Monitor::new();
        let mut rx = monitor.subscribe();
        let (sampler, _calls) = CountingSampler::new();

        let handle = monitor.spawn_sampler(sampler, Duration::from_secs(1));
        assert_eq!(rx.recv().await.unwrap().memory_used_bytes, 1);

        // An out-of-band event between ticks (Go's Monitor.Report).
        monitor.publish(Metrics {
            tokens_per_second: 42.0,
            ..Default::default()
        });
        assert_eq!(rx.recv().await.unwrap().tokens_per_second, 42.0);
        assert_eq!(rx.recv().await.unwrap().memory_used_bytes, 2);

        handle.stop().await;
    }

    #[test]
    fn memory_stats_map_to_used_and_available() {
        const GIB: u64 = 1 << 30;
        let m = metrics_from_memory_stats(veloxquant_system::MemoryStats {
            total_bytes: 16 * GIB,
            available_bytes: 6 * GIB,
        });
        assert_eq!(m.memory_used_bytes, 10 * GIB);
        assert_eq!(m.memory_available_bytes, 6 * GIB);
        // Inference-side fields are never fabricated by the system sampler.
        assert_eq!(m.tokens_per_second, 0.0);
        assert_eq!(m.time_to_first_token, Duration::ZERO);
        assert_eq!(m.context_length, 0);
        assert_eq!(m.kv_cache_bytes, 0);
        assert_eq!(m.compression_ratio, 0.0);

        // Unknown memory (both 0) stays 0 rather than underflowing, and a
        // reading where available exceeds total (two racing refreshes)
        // saturates instead of wrapping.
        let unknown = metrics_from_memory_stats(veloxquant_system::MemoryStats::default());
        assert_eq!(unknown.memory_used_bytes, 0);
        let racy = metrics_from_memory_stats(veloxquant_system::MemoryStats {
            total_bytes: GIB,
            available_bytes: 2 * GIB,
        });
        assert_eq!(racy.memory_used_bytes, 0);
    }

    #[tokio::test]
    async fn system_sampler_reports_live_host_memory() {
        let metrics = SystemSampler::new().sample().await.unwrap();
        let stats = veloxquant_system::memory_stats();
        // Two separate reads, so only check the relationships that can't
        // be broken by memory moving between them. `available` may
        // legitimately read 0 on macOS (see `metrics_from_memory_stats`).
        if stats.total_bytes > 0 {
            assert!(metrics.memory_used_bytes <= stats.total_bytes);
            assert!(metrics.memory_available_bytes <= stats.total_bytes);
        }
        assert_eq!(metrics.tokens_per_second, 0.0);
    }

    #[tokio::test]
    async fn system_sampler_drives_monitor_end_to_end() {
        let monitor = Monitor::new();
        let mut rx = monitor.subscribe();
        let handle = monitor.spawn_sampler(SystemSampler::new(), Duration::from_millis(20));

        for _ in 0..2 {
            let sample = tokio::time::timeout(Duration::from_secs(5), rx.recv())
                .await
                .expect("samples should keep arriving")
                .unwrap();
            let total = veloxquant_system::memory_stats().total_bytes;
            if total > 0 {
                assert!(sample.memory_used_bytes <= total);
            }
        }
        handle.stop().await;
        assert!(handle_is_stopped_after_stop(&monitor).await);
    }

    /// After `stop`, a fresh subscriber sees nothing further published.
    async fn handle_is_stopped_after_stop(monitor: &Monitor) -> bool {
        let mut rx = monitor.subscribe();
        tokio::time::sleep(Duration::from_millis(100)).await;
        matches!(rx.try_recv(), Err(broadcast::error::TryRecvError::Empty))
    }
}
