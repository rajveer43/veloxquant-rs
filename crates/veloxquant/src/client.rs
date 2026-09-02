//! The [`Client`] entry point and its [`ClientBuilder`].

use std::sync::Arc;
use std::time::Duration;

use veloxquant_core::{Config, OptimizationProfile, Result, DEFAULT_RUNTIME_URL, DEFAULT_TIMEOUT};
use veloxquant_memory::{MemoryService, OptimizationService};
use veloxquant_system::SystemService;

#[cfg(feature = "openai")]
use crate::chat::ChatApi;
use crate::models::ModelRegistry;
#[cfg(feature = "runtime")]
use veloxquant_runtime::RuntimeClient;

/// The VeloxQuant SDK entry point.
///
/// Construct via [`Client::builder`]. Cheap to clone — internally it's an
/// [`Arc`] over shared configuration and service handles, so it is safe and
/// idiomatic to share across tasks:
///
/// ```no_run
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// use std::sync::Arc;
/// use veloxquant::Client;
///
/// let client = Arc::new(Client::builder().auto_detect().build()?);
/// let client2 = client.clone();
/// tokio::spawn(async move {
///     let _ = client2.system().info().await;
/// });
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Client {
    inner: Arc<Inner>,
}

struct Inner {
    config: Config,
    system: SystemService,
    memory: MemoryService,
    optimize: OptimizationService,
    models: ModelRegistry,
    #[cfg(feature = "runtime")]
    runtime: RuntimeClient,
}

impl Client {
    /// Starts building a new [`Client`] with default configuration.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    /// Returns the system/hardware detection service.
    pub fn system(&self) -> &SystemService {
        &self.inner.system
    }

    /// Returns the memory estimation service.
    pub fn memory(&self) -> &MemoryService {
        &self.inner.memory
    }

    /// Returns the optimization recommendation service.
    pub fn optimize(&self) -> &OptimizationService {
        &self.inner.optimize
    }

    /// Returns the curated model registry.
    pub fn models(&self) -> &ModelRegistry {
        &self.inner.models
    }

    /// Returns a client for the VeloxQuant runtime's control endpoints
    /// (currently: health checks).
    #[cfg(feature = "runtime")]
    pub fn runtime(&self) -> &RuntimeClient {
        &self.inner.runtime
    }

    /// Returns a handle for issuing chat completions.
    #[cfg(feature = "openai")]
    pub fn chat(&self) -> Result<ChatApi> {
        ChatApi::new(
            self.inner.config.runtime_url.clone(),
            self.inner.config.timeout,
        )
    }

    /// The optimization profile this client will use by default.
    pub fn profile(&self) -> OptimizationProfile {
        self.inner.config.profile
    }

    /// The runtime URL this client is configured to talk to.
    pub fn runtime_url(&self) -> &str {
        &self.inner.config.runtime_url
    }
}

/// Builder for [`Client`].
#[derive(Debug, Clone)]
pub struct ClientBuilder {
    runtime_url: String,
    auto_detect: bool,
    profile: OptimizationProfile,
    timeout: Duration,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self {
            runtime_url: DEFAULT_RUNTIME_URL.to_string(),
            auto_detect: false,
            profile: OptimizationProfile::default(),
            timeout: DEFAULT_TIMEOUT,
        }
    }
}

impl ClientBuilder {
    /// Sets the VeloxQuant runtime's base URL. Defaults to
    /// `http://localhost:8765`.
    pub fn runtime_url(mut self, url: impl Into<String>) -> Self {
        self.runtime_url = url.into();
        self
    }

    /// Points this client at an OpenAI-compatible endpoint (e.g.
    /// `http://localhost:8765/v1`). Equivalent to [`ClientBuilder::runtime_url`];
    /// provided as a discoverable alias for OpenAI-compatibility use cases.
    pub fn openai_compatible(mut self, base_url: impl Into<String>) -> Self {
        self.runtime_url = base_url.into();
        self
    }

    /// Enables hardware auto-detection when the client is built. Currently
    /// this only affects whether [`Client::profile`] is seeded from
    /// detected hardware; per-request estimation always reflects live
    /// system state via [`Client::system`], regardless of this flag.
    pub fn auto_detect(mut self) -> Self {
        self.auto_detect = true;
        self
    }

    /// Sets the default optimization profile.
    pub fn profile(mut self, profile: OptimizationProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Sets the per-request timeout for runtime HTTP calls.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Builds the [`Client`].
    ///
    /// Fails only if the underlying HTTP client cannot be constructed
    /// (e.g. an invalid TLS configuration on the host).
    pub fn build(self) -> Result<Client> {
        let profile = if self.auto_detect {
            veloxquant_system::detect().recommended_profile
        } else {
            self.profile
        };

        let config = Config {
            runtime_url: self.runtime_url,
            auto_detect: self.auto_detect,
            profile,
            timeout: self.timeout,
        };

        #[cfg(feature = "runtime")]
        let runtime = RuntimeClient::new(config.runtime_url.clone(), config.timeout)?;

        Ok(Client {
            inner: Arc::new(Inner {
                system: SystemService::new(),
                memory: MemoryService::new(),
                optimize: OptimizationService::new(),
                models: ModelRegistry::new(),
                #[cfg(feature = "runtime")]
                runtime,
                config,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_builder_uses_default_runtime_url() {
        let client = Client::builder().build().unwrap();
        assert_eq!(client.runtime_url(), DEFAULT_RUNTIME_URL);
        assert_eq!(client.profile(), OptimizationProfile::Balanced);
    }

    #[test]
    fn custom_runtime_url_is_respected() {
        let client = Client::builder()
            .runtime_url("http://example.com:9000")
            .build()
            .unwrap();
        assert_eq!(client.runtime_url(), "http://example.com:9000");
    }

    #[test]
    fn explicit_profile_overrides_default() {
        let client = Client::builder()
            .profile(OptimizationProfile::Memory)
            .build()
            .unwrap();
        assert_eq!(client.profile(), OptimizationProfile::Memory);
    }

    #[test]
    fn client_is_cheaply_cloneable() {
        let client = Client::builder().build().unwrap();
        let clone = client.clone();
        assert_eq!(client.runtime_url(), clone.runtime_url());
    }
}
