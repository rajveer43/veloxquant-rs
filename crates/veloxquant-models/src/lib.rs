//! Local Hugging Face model cache management for the VeloxQuant Rust SDK.
//!
//! Mirrors `@veloxquant/sdk`'s `src/localModels.ts`: lists, pulls, and
//! deletes model weights already (or about to be) downloaded to the local
//! Hugging Face cache, by shelling out to a short Python snippet that uses
//! `huggingface_hub`'s own `scan_cache_dir()` / `snapshot_download()` /
//! `delete_revisions()` APIs rather than reimplementing the cache's
//! content-addressed blob/symlink layout natively — that layout is owned
//! and evolved by `huggingface_hub` itself, so a native Rust
//! reimplementation would silently drift out of sync with upstream
//! cache-format changes.
//!
//! This is distinct from `veloxquant::Client::models()`, which lists the
//! curated *compression methods / model registry*, not locally downloaded
//! weights.

use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::process::Command;

use veloxquant_core::VeloxQuantError;

/// A Python interpreter used to run the `huggingface_hub` cache-management
/// snippets.
///
/// Defaults to `python3` on `$PATH`. Construct with
/// [`PythonInterpreter::new`] to point at a specific interpreter (e.g. a
/// virtualenv's `python`).
#[derive(Debug, Clone)]
pub struct PythonInterpreter {
    path: PathBuf,
}

impl Default for PythonInterpreter {
    fn default() -> Self {
        Self {
            path: PathBuf::from("python3"),
        }
    }
}

impl PythonInterpreter {
    /// Creates a [`PythonInterpreter`] pointing at the given interpreter
    /// executable (a bare name resolved via `$PATH`, or an absolute path).
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The interpreter executable this instance will invoke.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

/// A model's weights present in the local Hugging Face cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalModel {
    /// The Hugging Face repo id (e.g. `"mlx-community/Qwen3-8B-4bit"`).
    pub id: String,
    /// Total size on disk, in bytes (resolves shared/symlinked blobs once,
    /// matching `scan_cache_dir()`'s own accounting).
    pub size_bytes: u64,
    /// When this model was last accessed, if `huggingface_hub` reported a
    /// timestamp.
    pub last_accessed: Option<SystemTime>,
}

/// Result of a successful [`pull_local_model`] call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PullModelResult {
    /// The model id that was downloaded.
    pub id: String,
    /// Total size on disk after the download, in bytes.
    pub size_bytes: u64,
}

/// Result of a successful [`delete_local_model`] call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeleteModelResult {
    /// The model id that was deleted.
    pub id: String,
    /// Bytes freed on disk by the deletion.
    pub freed_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct RawLocalModel {
    id: String,
    size_bytes: u64,
    last_accessed: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ScanResponse {
    #[serde(default)]
    repos: Vec<RawLocalModel>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PullResponse {
    id: Option<String>,
    size_bytes: Option<u64>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DeleteResponse {
    id: Option<String>,
    freed_bytes: Option<u64>,
    error: Option<String>,
}

/// Mirrors `localModels.ts`'s `SCAN_CACHE_SNIPPET`: uses
/// `huggingface_hub.scan_cache_dir()`, treating a `CacheNotFound` error as
/// an empty list rather than a hard failure.
const SCAN_CACHE_SNIPPET: &str = r#"
import json
try:
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound
except ImportError:
    print(json.dumps({"error": "huggingface_hub is not importable"}))
else:
    try:
        info = scan_cache_dir()
    except CacheNotFound:
        print(json.dumps({"repos": []}))
    else:
        repos = [
            {
                "id": repo.repo_id,
                "size_bytes": repo.size_on_disk,
                "last_accessed": repo.last_accessed,
            }
            for repo in info.repos
            if repo.repo_type == "model"
        ]
        print(json.dumps({"repos": repos}))
"#;

/// Mirrors `localModels.ts`'s pull snippet: `snapshot_download()` followed
/// by a `scan_cache_dir()` re-scan to report the resulting size. The model
/// id is read from `sys.argv[1]`, never interpolated into this source
/// string, matching `localModels.ts:130-135`'s command-injection
/// mitigation.
const PULL_SNIPPET: &str = r#"
import json, sys
try:
    from huggingface_hub import snapshot_download, scan_cache_dir
    from huggingface_hub.errors import CacheNotFound
except ImportError:
    print(json.dumps({"error": "huggingface_hub is not importable"}))
else:
    model_id = sys.argv[1]
    try:
        snapshot_download(repo_id=model_id)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
    else:
        try:
            info = scan_cache_dir()
        except CacheNotFound:
            size_bytes = 0
        else:
            repo = next((r for r in info.repos if r.repo_type == "model" and r.repo_id == model_id), None)
            size_bytes = repo.size_on_disk if repo is not None else 0
        print(json.dumps({"id": model_id, "size_bytes": size_bytes}))
"#;

/// Mirrors `localModels.ts`'s delete snippet: scan -> find the repo ->
/// `delete_revisions()` -> `.execute()`, rather than an `rm -rf` on a
/// resolved path (the cache's blob layout is content-addressed and shared
/// across revisions/repos via symlinks, so a naive recursive delete risks
/// corrupting a *different* cached model's blobs).
const DELETE_SNIPPET: &str = r#"
import json, sys
try:
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound
except ImportError:
    print(json.dumps({"error": "huggingface_hub is not importable"}))
else:
    model_id = sys.argv[1]
    try:
        info = scan_cache_dir()
    except CacheNotFound:
        print(json.dumps({"error": f"No cached model found with id {model_id!r}"}))
    else:
        repo = next((r for r in info.repos if r.repo_type == "model" and r.repo_id == model_id), None)
        if repo is None:
            print(json.dumps({"error": f"No cached model found with id {model_id!r}"}))
        else:
            revisions = [rev.commit_hash for rev in repo.revisions]
            strategy = info.delete_revisions(*revisions)
            strategy.execute()
            print(json.dumps({"id": model_id, "freed_bytes": strategy.expected_freed_size}))
"#;

/// Runs `interpreter -c <snippet> [extra_args...]` and returns captured
/// stdout as a `String`. `extra_args` are passed as separate argv elements
/// (never interpolated into `snippet`), so a value containing shell
/// metacharacters can't be interpreted as shell syntax — this is the same
/// command-injection mitigation `localModels.ts:130-135` documents.
///
/// An optional `timeout` bounds the wait; `None` waits indefinitely
/// (used for `pull`, since downloads can take many minutes).
async fn run_snippet(
    interpreter: &PythonInterpreter,
    snippet: &str,
    extra_args: &[&str],
    timeout: Option<Duration>,
) -> Result<String, String> {
    let mut command = Command::new(interpreter.path());
    command
        .arg("-c")
        .arg(snippet)
        .args(extra_args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let run = async {
        let output = command
            .output()
            .await
            .map_err(|e| format!("could not launch {:?}: {e}", interpreter.path()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            return Err(if stderr.is_empty() {
                format!("interpreter exited with status {}", output.status)
            } else {
                stderr
            });
        }

        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    };

    match timeout {
        Some(duration) => tokio::time::timeout(duration, run)
            .await
            .map_err(|_| "timed out".to_string())?,
        None => run.await,
    }
}

/// Default timeout for `list`/`delete` calls (which are fast, in-process
/// cache scans). `pull` has no default timeout — see [`pull_local_model`].
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Lists model weights already downloaded to the local Hugging Face cache
/// (read-only — no deletion/pull support here).
///
/// Distinct from `veloxquant::Client::models().list()`, which lists the
/// curated *compression methods* registry, not downloaded weights.
///
/// A missing local cache (`CacheNotFound`) is reported as an empty list,
/// not an error, matching `localModels.ts:56-83`.
pub async fn list_local_models(
    python: &PythonInterpreter,
) -> Result<Vec<LocalModel>, VeloxQuantError> {
    let stdout = run_snippet(python, SCAN_CACHE_SNIPPET, &[], Some(DEFAULT_TIMEOUT))
        .await
        .map_err(VeloxQuantError::ModelScanFailed)?;

    let parsed: ScanResponse =
        serde_json::from_str(&stdout).map_err(VeloxQuantError::Serialization)?;

    if let Some(error) = parsed.error {
        return Err(VeloxQuantError::ModelScanFailed(error));
    }

    Ok(parsed
        .repos
        .into_iter()
        .map(|repo| LocalModel {
            id: repo.id,
            size_bytes: repo.size_bytes,
            last_accessed: repo
                .last_accessed
                .map(|secs| UNIX_EPOCH + Duration::from_secs_f64(secs)),
        })
        .collect())
}

/// Downloads a model's weights into the local Hugging Face cache via
/// `snapshot_download()` — the same library [`list_local_models`] reads
/// back from — without loading it into an inference runtime.
///
/// No progress callback: `snapshot_download()`'s tqdm-based progress
/// doesn't cross a subprocess stdout boundary cleanly, matching
/// `localModels.ts:93-98`'s documented reasoning.
///
/// **No default timeout** — downloads can take many minutes for large
/// models, so this deliberately waits indefinitely for the subprocess to
/// finish rather than reusing a short RPC-style timeout. `model_id` is
/// passed as its own argv element, never interpolated into the Python
/// source, so an id containing shell metacharacters can't be interpreted
/// as shell syntax.
pub async fn pull_local_model(
    python: &PythonInterpreter,
    model_id: &str,
) -> Result<PullModelResult, VeloxQuantError> {
    let stdout = run_snippet(python, PULL_SNIPPET, &[model_id], None)
        .await
        .map_err(|detail| VeloxQuantError::ModelPullFailed {
            id: model_id.to_string(),
            detail,
        })?;

    let parsed: PullResponse =
        serde_json::from_str(&stdout).map_err(VeloxQuantError::Serialization)?;

    if let Some(error) = parsed.error {
        return Err(VeloxQuantError::ModelPullFailed {
            id: model_id.to_string(),
            detail: error,
        });
    }

    Ok(PullModelResult {
        id: parsed.id.unwrap_or_else(|| model_id.to_string()),
        size_bytes: parsed.size_bytes.unwrap_or(0),
    })
}

/// Deletes a model's weights from the local Hugging Face cache using
/// `huggingface_hub`'s own eviction API (scan -> `delete_revisions()` ->
/// `.execute()`) rather than an `rm -rf` on a resolved path — the cache's
/// blob layout is content-addressed and shared across revisions/repos via
/// symlinks, so a naive recursive delete risks corrupting a *different*
/// cached model's blobs. Matches `localModels.ts:159-207`.
pub async fn delete_local_model(
    python: &PythonInterpreter,
    model_id: &str,
) -> Result<DeleteModelResult, VeloxQuantError> {
    let stdout = run_snippet(python, DELETE_SNIPPET, &[model_id], Some(DEFAULT_TIMEOUT))
        .await
        .map_err(|detail| VeloxQuantError::ModelDeleteFailed {
            id: model_id.to_string(),
            detail,
        })?;

    let parsed: DeleteResponse =
        serde_json::from_str(&stdout).map_err(VeloxQuantError::Serialization)?;

    if let Some(error) = parsed.error {
        return Err(VeloxQuantError::ModelDeleteFailed {
            id: model_id.to_string(),
            detail: error,
        });
    }

    Ok(DeleteModelResult {
        id: parsed.id.unwrap_or_else(|| model_id.to_string()),
        freed_bytes: parsed.freed_bytes.unwrap_or(0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake "interpreter" is just another Python-shaped executable: we
    /// point `PythonInterpreter` at a tiny shell/python fixture script
    /// rather than inventing a new mocking seam, matching how these tests
    /// need to exercise the real `Command`-spawning path end to end without
    /// depending on `huggingface_hub` being installed in CI.
    ///
    /// Since a real Python interpreter may not be available in the test
    /// environment, these fixture scripts are themselves tiny Python
    /// scripts invoked via `python3 -c`, and we build a fake "interpreter"
    /// as a small wrapper script that ignores the injected snippet and
    /// prints canned JSON — this proves argv-passing (never shell
    /// interpolation) and JSON-parsing/error-mapping without requiring
    /// `huggingface_hub` to be installed.
    fn fixture_interpreter(dir: &std::path::Path, script: &str) -> PythonInterpreter {
        let path = dir.join("fake_python.sh");
        std::fs::write(&path, script).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&path).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&path, perms).unwrap();
        }
        PythonInterpreter::new(path)
    }

    #[tokio::test]
    async fn list_local_models_parses_repos() {
        let dir = tempdir();
        let interpreter = fixture_interpreter(
            dir.path(),
            "#!/bin/sh\necho '{\"repos\":[{\"id\":\"org/model\",\"size_bytes\":123,\"last_accessed\":1700000000.0}]}'\n",
        );

        let models = list_local_models(&interpreter).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "org/model");
        assert_eq!(models[0].size_bytes, 123);
        assert!(models[0].last_accessed.is_some());
    }

    #[tokio::test]
    async fn list_local_models_empty_repos_is_empty_list() {
        let dir = tempdir();
        let interpreter = fixture_interpreter(dir.path(), "#!/bin/sh\necho '{\"repos\":[]}'\n");

        let models = list_local_models(&interpreter).await.unwrap();
        assert!(models.is_empty());
    }

    #[tokio::test]
    async fn list_local_models_error_field_surfaces_as_scan_failed() {
        let dir = tempdir();
        let interpreter = fixture_interpreter(
            dir.path(),
            "#!/bin/sh\necho '{\"error\":\"huggingface_hub is not importable\"}'\n",
        );

        let err = list_local_models(&interpreter).await.unwrap_err();
        assert!(matches!(err, VeloxQuantError::ModelScanFailed(_)));
        assert!(err
            .to_string()
            .contains("huggingface_hub is not importable"));
    }

    #[tokio::test]
    async fn pull_local_model_passes_model_id_as_argv_not_shell_string() {
        let dir = tempdir();
        // Echo argv[1] (the model id, after `-c <snippet>`) back in the
        // JSON, proving the id arrives as a separate argv element rather
        // than being concatenated into the snippet / a shell string. A
        // shell-metacharacter-laden id would break a naive
        // string-interpolation implementation but must round-trip cleanly
        // here.
        let interpreter = fixture_interpreter(
            dir.path(),
            r#"#!/bin/sh
# $1 is "-c", $2 is the snippet, $3 is the model id.
printf '{"id":"%s","size_bytes":42}' "$3"
"#,
        );

        let result = pull_local_model(&interpreter, "org/model; rm -rf /")
            .await
            .unwrap();
        assert_eq!(result.id, "org/model; rm -rf /");
        assert_eq!(result.size_bytes, 42);
    }

    #[tokio::test]
    async fn pull_local_model_error_field_surfaces_as_pull_failed_with_id() {
        let dir = tempdir();
        let interpreter = fixture_interpreter(
            dir.path(),
            "#!/bin/sh\necho '{\"error\":\"404 not found\"}'\n",
        );

        let err = pull_local_model(&interpreter, "org/missing")
            .await
            .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("org/missing"));
        assert!(message.contains("404 not found"));
    }

    #[tokio::test]
    async fn delete_local_model_success() {
        let dir = tempdir();
        let interpreter = fixture_interpreter(
            dir.path(),
            "#!/bin/sh\necho '{\"id\":\"org/model\",\"freed_bytes\":999}'\n",
        );

        let result = delete_local_model(&interpreter, "org/model").await.unwrap();
        assert_eq!(result.id, "org/model");
        assert_eq!(result.freed_bytes, 999);
    }

    #[tokio::test]
    async fn delete_local_model_not_found_surfaces_as_delete_failed() {
        let dir = tempdir();
        let interpreter = fixture_interpreter(
            dir.path(),
            "#!/bin/sh\necho '{\"error\":\"No cached model found with id \\'org/missing\\'\"}'\n",
        );

        let err = delete_local_model(&interpreter, "org/missing")
            .await
            .unwrap_err();
        assert!(matches!(err, VeloxQuantError::ModelDeleteFailed { .. }));
    }

    #[tokio::test]
    async fn interpreter_not_found_surfaces_clear_error() {
        let interpreter = PythonInterpreter::new("/nonexistent/definitely/not/python");
        let err = list_local_models(&interpreter).await.unwrap_err();
        assert!(matches!(err, VeloxQuantError::ModelScanFailed(_)));
    }

    /// Minimal tempdir helper (avoids adding the `tempfile` crate as a new
    /// dependency for a handful of tests).
    struct TempDir(PathBuf);
    impl TempDir {
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn tempdir() -> TempDir {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let mut dir = std::env::temp_dir();
        let unique = format!(
            "veloxquant-models-test-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        dir.push(unique);
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
}
