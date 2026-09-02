//! AutoPilot (automatic model/context/compression selection and a ready
//! chat session) is planned for v0.3.0 and is not implemented in this
//! release.
//!
//! Track progress at
//! <https://github.com/rajveer43/veloxquant-rs/issues> (see the v0.3.0
//! milestone). In the meantime, `Client::system`, `Client::memory`, and
//! `Client::optimize` can be combined manually — see `examples/chat.rs`
//! and the README's "Optimization Profiles" section.

fn main() {
    eprintln!(
        "AutoPilot is not yet implemented; combine Client::system/memory/optimize manually for now"
    );
    std::process::exit(1);
}
