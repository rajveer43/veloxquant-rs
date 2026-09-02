//! Shared error types, configuration, and HTTP client scaffolding for the
//! VeloxQuant Rust SDK.
//!
//! This crate is an internal building block of the [`veloxquant`] facade
//! crate and is not usually depended on directly by applications.
//!
//! [`veloxquant`]: https://docs.rs/veloxquant

pub mod client;
pub mod config;
pub mod error;
pub mod types;

pub use client::build_http_client;
pub use config::{Config, DEFAULT_RUNTIME_URL, DEFAULT_TIMEOUT};
pub use error::{Result, VeloxQuantError};
pub use types::{format_bytes, OptimizationProfile};
